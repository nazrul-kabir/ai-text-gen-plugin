// server.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const { pipeline, env } = require('@xenova/transformers');

const app = express();
const PORT = process.env.PORT || 3005;

// Configure transformers environment
env.localModelPath = './models/';
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.backends.onnx.wasm.numThreads = 4; // Use multiple threads
env.backends.onnx.wasm.simd = true; 

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static HTML/CSS/JS files

// Global model variable - loaded once and reused
let generator = null;
let modelLoading = false;

// Model loading function
async function loadModel() {
  if (generator) return generator;
  if (modelLoading) {
    // Wait for existing loading to complete
    while (modelLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return generator;
  }

  modelLoading = true;
  console.log('Loading DistilGPT2 model...');
  
  try {
    // Load with optimizations
    generator = await pipeline('text-generation', 'distilgpt2', {
      quantized: true,
      device: 'cpu'
    });
    console.log('✅ Model loaded successfully');
    modelLoading = false;
    return generator;
  } catch (error) {
    console.error('❌ Model loading failed:', error);
    modelLoading = false;
    throw error;
  }
}

// Optimized generation with caching and batching
const generationCache = new Map();
const MAX_CACHE_SIZE = 100;

function getCacheKey(topic, count) {
  return `${topic.toLowerCase().trim()}_${count}`;
}

function cleanCache() {
  if (generationCache.size > MAX_CACHE_SIZE) {
    const firstKey = generationCache.keys().next().value;
    generationCache.delete(firstKey);
  }
}

// Helper functions (same logic as browser version)
function createStructuredPrompt(topic, count) {
  let prompt = `Here are ${count} key facts about ${topic}:\n\n`;
  
  for (let i = 1; i <= count; i++) {
    if (i === 1) {
      prompt += `${i}. ${topic} is important because it`;
    } else if (i === 2) {
      prompt += `\n${i}. Research in ${topic} has shown that`;
    } else {
      prompt += `\n${i}. The benefits of ${topic} include`;
    }
  }
  
  return prompt;
}

async function generateSinglePoint(topic, pointNumber) {
  const starters = [
    `${topic} is beneficial because it`,
    `Research shows that ${topic}`,
    `One advantage of ${topic} is that it`,
    `Studies indicate that ${topic}`,
    `The importance of ${topic} lies in how it`,
    `Experts agree that ${topic}`,
    `${topic} has been proven to`
  ];
  
  const starter = starters[pointNumber % starters.length];
  
  const result = await generator(starter, {
    max_new_tokens: 25,
    temperature: 0.8,
    do_sample: true,
    top_p: 0.9,
    top_k: 40,
    repetition_penalty: 1.1,
    return_full_text: false,
    pad_token_id: 50256,
    eos_token_id: 50256,
    // Performance optimizations
      use_cache: true,
      output_scores: false,
      output_attentions: false
  });
  
  let point = result[0].generated_text.trim();
  const sentences = point.split(/[.!?]/);
  if (sentences.length > 0) {
    point = sentences[0].trim();
    if (!point.match(/[.!?]$/)) {
      point += '.';
    }
  }
  
  return starter.replace(`${topic} is beneficial because it`, `${topic}`) + ' ' + point;
}

function parseAndExtractPoints(text, expectedCount) {
  let points = [];
  
  // Split by numbers
  const numberedSections = text.split(/\d+\./);
  if (numberedSections.length > 1) {
    for (let i = 1; i < numberedSections.length; i++) {
      const section = numberedSections[i].trim();
      if (section.length > 10) {
        const cleanPoint = cleanUpPoint(section);
        if (cleanPoint) points.push(cleanPoint);
      }
    }
  }
  
  // Split by bullet points if not enough
  if (points.length < expectedCount) {
    const bulletSections = text.split(/[•\-\*]/);
    if (bulletSections.length > 1) {
      for (let i = 1; i < bulletSections.length; i++) {
        const section = bulletSections[i].trim();
        if (section.length > 10) {
          const cleanPoint = cleanUpPoint(section);
          if (cleanPoint) points.push(cleanPoint);
        }
      }
    }
  }
  
  return points;
}

function cleanUpPoint(text) {
  text = text.replace(/^(is that|that|because|by|through)\s+/i, '');
  text = text.replace(/^\s*[•\-\*]\s*/, '');
  text = text.replace(/^\d+\.\s*/, '');
  text = text.trim();
  
  if (text.length < 10) return null;
  
  const sentences = text.split(/[.!?]/);
  if (sentences.length > 0) {
    let cleanText = sentences[0].trim();
    if (cleanText.length > 10) {
      cleanText = cleanText.charAt(0).toUpperCase() + cleanText.slice(1);
      if (!cleanText.match(/[.!?]$/)) {
        cleanText += '.';
      }
      return cleanText;
    }
  }
  
  return null;
}

async function generateCompleteSet(topic, requestedCount) {
  const maxAttempts = 2; // Fewer attempts needed with better performance
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const structuredPrompt = createStructuredPrompt(topic, requestedCount);
      
      const result = await generator(structuredPrompt, {
        max_new_tokens: Math.min(150, requestedCount * 40),
        temperature: 0.7 + (attempt * 0.1),
        do_sample: true,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.2 + (attempt * 0.1),
        return_full_text: false,
        pad_token_id: 50256,
        eos_token_id: 50256
      });
      
      const points = parseAndExtractPoints(result[0].generated_text, requestedCount);
      
      if (points.length >= requestedCount) {
        return points.slice(0, requestedCount);
      }
      
      // Fill gap if partially successful
      if (points.length > 0 && points.length < requestedCount) {
        const needed = requestedCount - points.length;
        for (let i = 0; i < needed; i++) {
          try {
            const additionalPoint = await generateSinglePoint(topic, points.length + i);
            points.push(additionalPoint);
          } catch (error) {
            console.warn(`Failed to generate additional point ${i + 1}:`, error);
          }
        }
        
        if (points.length >= requestedCount) {
          return points.slice(0, requestedCount);
        }
      }
      
    } catch (error) {
      console.warn(`Attempt ${attempt + 1} failed:`, error);
      continue;
    }
  }
  
  // Final fallback - generate individual points
  const points = [];
  for (let i = 0; i < requestedCount; i++) {
    try {
      const point = await generateSinglePoint(topic, i);
      points.push(point);
    } catch (error) {
      points.push(`${topic} is an important topic that deserves consideration.`);
    }
  }
  
  return points;
}

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/status', async (req, res) => {
  try {
    if (!generator && !modelLoading) {
      res.json({ status: 'not_loaded', message: 'Model not loaded' });
    } else if (modelLoading) {
      res.json({ status: 'loading', message: 'Model is loading...' });
    } else {
      res.json({ status: 'ready', message: 'Model is ready' });
    }
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, count = 3 } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' });
    }
    await loadModel();
    const requestedCount = Math.min(Math.max(count, 1), 5);
    const cacheKey = getCacheKey(prompt, requestedCount);
    if (generationCache.has(cacheKey)) {
      console.log('Serving from cache:', cacheKey);
      const cached = generationCache.get(cacheKey);
      return res.json({
        success: true,
        topic: prompt,
        requestedCount,
        generatedCount: cached.points.length,
        points: cached.points,
        generationTime: cached.generationTime,
        cached: true
      });
    }
    console.log(`Generating ${requestedCount} bullet points about: ${prompt}`);
    const startTime = Date.now();
    const points = await generateCompleteSet(prompt, requestedCount);
    const generationTime = Date.now() - startTime;
    cleanCache();
    generationCache.set(cacheKey, { points, generationTime });
    console.log(`✅ Generated ${points.length} points in ${generationTime}ms`);
    res.json({
      success: true,
      topic: prompt,
      requestedCount,
      generatedCount: points.length,
      points: points,
      generationTime: generationTime,
      cached: false
    });
  } catch (error) {
    console.error('Generation error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// Initialize model on startup
loadModel().catch(console.error);

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log('📝 Bullet point generator API ready!');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down gracefully...');
  process.exit(0);
});
module.exports = app;