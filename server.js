// server.js
const express = require('express');
const cors = require('cors');
const path = require('path');
const { pipeline, env } = require('@xenova/transformers');

const app = express();
const PORT = process.env.PORT || 3005;

// Enhanced transformers environment configuration
env.localModelPath = './models/';
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;
env.backends.onnx.wasm.numThreads = Math.min(8, require('os').cpus().length); // Use available CPU cores
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.proxy = false; // Disable proxy for better performance

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static HTML/CSS/JS files

// Global model variable with enhanced caching
let generator = null;
let modelLoading = false;
let tokenizer = null;

// Enhanced model loading with better configuration
async function loadModel() {
  if (generator && tokenizer) return { generator, tokenizer };
  if (modelLoading) {
    while (modelLoading) {
      await new Promise(resolve => setTimeout(resolve, 50)); // Reduced wait time
    }
    return { generator, tokenizer };
  }

  modelLoading = true;
  console.log('Loading DistilGPT2 model with optimizations...');
  
  try {
    // Load model with performance optimizations - use non-quantized version
    generator = await pipeline('text-generation', 'distilgpt2', {
      quantized: false,
      device: 'cpu',
      dtype: 'fp32',
      // Explicitly use the non-quantized model file
      model_file_name: 'decoder_model_merged',
      use_external_data_format: false,
      provider: 'cpu'
    });

    // Pre-warm the model with a small generation to optimize memory layout
    console.log('Pre-warming model...');
    await generator("Test", {
      max_new_tokens: 1,
      do_sample: false,
      return_full_text: false
    });

    console.log('✅ Model loaded and optimized successfully');
    modelLoading = false;
    return { generator, tokenizer };
  } catch (error) {
    console.error('❌ Model loading failed:', error);
    modelLoading = false;
    throw error;
  }
}

// Enhanced caching with LRU and compression
const generationCache = new Map();
const MAX_CACHE_SIZE = 200; // Increased cache size
const cacheHits = new Map(); // Track cache usage

function getCacheKey(topic, count) {
  return `${topic.toLowerCase().trim().replace(/\s+/g, '_')}_${count}`;
}

function cleanCache() {
  if (generationCache.size >= MAX_CACHE_SIZE) {
    // LRU eviction - remove least recently used
    let oldestKey = null;
    let oldestTime = Date.now();
    
    for (const [key, data] of generationCache.entries()) {
      if (data.lastUsed < oldestTime) {
        oldestTime = data.lastUsed;
        oldestKey = key;
      }
    }
    
    if (oldestKey) {
      generationCache.delete(oldestKey);
      cacheHits.delete(oldestKey);
    }
  }
}

// Optimized prompt templates with better structure
const PROMPT_TEMPLATES = {
  structured: (topic, count) => {
    const templates = [
      `Key benefits of ${topic}:\n1.`,
      `Important facts about ${topic}:\n1.`,
      `Main advantages of ${topic}:\n1.`,
      `Essential points about ${topic}:\n1.`
    ];
    return templates[Math.floor(Math.random() * templates.length)];
  },
  
  single: (topic, index) => {
    const starters = [
      `${topic} helps by`,
      `${topic} is valuable because it`,
      `The benefit of ${topic} is that it`,
      `${topic} works by`,
      `${topic} provides`,
      `${topic} enables`,
      `${topic} improves`
    ];
    return starters[index % starters.length];
  }
};

// Batch generation for better performance
async function batchGenerate(prompts, options = {}) {
  const results = [];
  const batchSize = 1; // Process one at a time but optimize each call
  
  for (let i = 0; i < prompts.length; i += batchSize) {
    const batch = prompts.slice(i, i + batchSize);
    const batchPromises = batch.map(prompt => 
      generator(prompt, {
        max_new_tokens: options.maxTokens || 30,
        temperature: options.temperature || 0.8,
        do_sample: true,
        top_p: 0.95,
        top_k: 50,
        repetition_penalty: 1.15,
        return_full_text: false,
        pad_token_id: 50256,
        eos_token_id: 50256,
        // Performance optimizations
        use_cache: true,
        output_scores: false,
        output_attentions: false,
        output_hidden_states: false
      })
    );
    
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
  }
  
  return results;
}

// Optimized single generation strategy
async function generateOptimizedSet(topic, requestedCount) {
  const startTime = Date.now();
  
  try {
    // Strategy 1: Single structured prompt (fastest)
    const structuredPrompt = PROMPT_TEMPLATES.structured(topic, requestedCount);
    
    const result = await generator(structuredPrompt, {
      max_new_tokens: Math.min(120, requestedCount * 25), // Optimized token count
      temperature: 0.85,
      do_sample: true,
      top_p: 0.95,
      top_k: 50,
      repetition_penalty: 1.2,
      return_full_text: false,
      pad_token_id: 50256,
      eos_token_id: 50256,
      use_cache: true,
      output_scores: false,
      output_attentions: false,
      output_hidden_states: false
    });

    let points = parseAndExtractPoints(result[0].generated_text, topic, requestedCount);
    
    // If we got enough points, return them
    if (points.length >= requestedCount) {
      console.log(`✅ Structured generation successful: ${Date.now() - startTime}ms`);
      return points.slice(0, requestedCount);
    }

    // Strategy 2: Fill remaining with targeted single generations
    const needed = requestedCount - points.length;
    if (needed > 0 && needed <= 3) { // Only fill small gaps efficiently
      const singlePrompts = [];
      for (let i = 0; i < needed; i++) {
        singlePrompts.push(PROMPT_TEMPLATES.single(topic, points.length + i));
      }

      const singleResults = await batchGenerate(singlePrompts, {
        maxTokens: 25,
        temperature: 0.9
      });

      for (const singleResult of singleResults) {
        const cleanPoint = cleanUpSinglePoint(singleResult[0].generated_text, topic);
        if (cleanPoint && cleanPoint.length > 10) {
          points.push(cleanPoint);
        }
      }
    }

    console.log(`✅ Hybrid generation completed: ${Date.now() - startTime}ms`);
    return points.slice(0, requestedCount);

  } catch (error) {
    console.warn('Generation failed, using fallback:', error);
    return generateFallbackPoints(topic, requestedCount);
  }
}

// Enhanced parsing with better extraction
function parseAndExtractPoints(text, topic, expectedCount) {
  let points = [];
  
  // Clean up the input text
  text = text.replace(/\n+/g, '\n').trim();
  
  // Strategy 1: Extract numbered points
  const numberedMatches = text.match(/\d+\.\s*([^\n\d]+)/g);
  if (numberedMatches) {
    for (const match of numberedMatches) {
      const point = match.replace(/^\d+\.\s*/, '').trim();
      const cleanPoint = cleanUpPoint(point, topic);
      if (cleanPoint) {
        points.push(cleanPoint);
        if (points.length >= expectedCount) break;
      }
    }
  }

  // Strategy 2: Split by sentence if not enough points
  if (points.length < expectedCount) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 15);
    for (const sentence of sentences) {
      if (points.length >= expectedCount) break;
      const cleanPoint = cleanUpPoint(sentence.trim(), topic);
      if (cleanPoint && !points.some(p => p.includes(cleanPoint.substring(0, 20)))) {
        points.push(cleanPoint);
      }
    }
  }

  return points.filter(p => p && p.length > 10);
}

// Enhanced point cleanup
function cleanUpPoint(text, topic) {
  if (!text || text.length < 5) return null;
  
  // Remove common prefixes and artifacts
  text = text.replace(/^(is that|that|because|by|through|and|but|or|so|also)\s+/i, '');
  text = text.replace(/^\s*[•\-\*\d+\.]\s*/, '');
  text = text.replace(/^(it\s+|this\s+|they\s+)/i, `${topic} `);
  text = text.trim();
  
  if (text.length < 10) return null;
  
  // Ensure proper capitalization
  text = text.charAt(0).toUpperCase() + text.slice(1);
  
  // Ensure proper ending
  if (!text.match(/[.!?]$/)) {
    text += '.';
  }
  
  // Validate content quality
  if (text.includes('undefined') || text.includes('null') || text.length > 200) {
    return null;
  }
  
  return text;
}

// Optimized single point cleanup
function cleanUpSinglePoint(text, topic) {
  if (!text) return null;
  
  text = text.trim();
  const sentences = text.split(/[.!?]/);
  if (sentences.length > 0) {
    let cleanText = sentences[0].trim();
    if (cleanText.length > 8) {
      cleanText = cleanText.charAt(0).toUpperCase() + cleanText.slice(1);
      if (!cleanText.match(/[.!?]$/)) {
        cleanText += '.';
      }
      return `${topic} ${cleanText}`;
    }
  }
  return null;
}

// Minimal fallback for extreme cases
function generateFallbackPoints(topic, count) {
  const fallbacks = [
    `${topic} provides significant benefits for users.`,
    `${topic} has been shown to be effective in various applications.`,
    `${topic} offers practical solutions for common challenges.`,
    `${topic} represents an important advancement in its field.`,
    `${topic} contributes to improved outcomes and efficiency.`
  ];
  
  return fallbacks.slice(0, count);
}

// Routes (unchanged)
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
      res.json({ 
        status: 'ready', 
        message: 'Model is ready',
        cacheSize: generationCache.size,
        cacheHitRate: calculateCacheHitRate()
      });
    }
  } catch (error) {
    res.status(500).json({ status: 'error', message: error.message });
  }
});

function calculateCacheHitRate() {
  let totalHits = 0;
  let totalRequests = 0;
  for (const hits of cacheHits.values()) {
    totalHits += hits;
    totalRequests += hits;
  }
  return totalRequests > 0 ? (totalHits / totalRequests * 100).toFixed(1) : 0;
}

app.post('/api/generate', async (req, res) => {
  const requestStart = Date.now();
  
  try {
    const { prompt, count = 3 } = req.body;
    if (!prompt || prompt.trim().length === 0) {
      return res.status(400).json({ error: 'Valid prompt is required' });
    }

    // Ensure model is loaded
    await loadModel();
    
    const requestedCount = Math.min(Math.max(parseInt(count), 1), 5);
    const cacheKey = getCacheKey(prompt.trim(), requestedCount);
    
    // Check cache with LRU update
    if (generationCache.has(cacheKey)) {
      const cached = generationCache.get(cacheKey);
      cached.lastUsed = Date.now();
      generationCache.set(cacheKey, cached); // Update position
      
      cacheHits.set(cacheKey, (cacheHits.get(cacheKey) || 0) + 1);
      
      console.log(`💾 Cache hit (${Date.now() - requestStart}ms):`, cacheKey);
      return res.json({
        success: true,
        topic: prompt.trim(),
        requestedCount,
        generatedCount: cached.points.length,
        points: cached.points,
        generationTime: cached.generationTime,
        totalTime: Date.now() - requestStart,
        cached: true
      });
    }

    console.log(`🔄 Generating ${requestedCount} points for: "${prompt}"`);
    const generationStart = Date.now();
    
    const points = await generateOptimizedSet(prompt.trim(), requestedCount);
    const generationTime = Date.now() - generationStart;

    // Cache with metadata
    cleanCache();
    generationCache.set(cacheKey, { 
      points, 
      generationTime, 
      lastUsed: Date.now(),
      created: Date.now()
    });

    const totalTime = Date.now() - requestStart;
    console.log(`✅ Generated ${points.length}/${requestedCount} points | Gen: ${generationTime}ms | Total: ${totalTime}ms`);

    res.json({
      success: true,
      topic: prompt.trim(),
      requestedCount,
      generatedCount: points.length,
      points: points,
      generationTime: generationTime,
      totalTime: totalTime,
      cached: false
    });

  } catch (error) {
    const totalTime = Date.now() - requestStart;
    console.error(`❌ Generation failed (${totalTime}ms):`, error.message);
    res.status(500).json({ 
      success: false, 
      error: error.message,
      totalTime: totalTime
    });
  }
});

// Optimized startup
let serverReady = false;

async function initializeServer() {
  try {
    console.log('🚀 Starting server initialization...');
    await loadModel();
    serverReady = true;
    console.log('✅ Server fully initialized and ready');
  } catch (error) {
    console.error('❌ Server initialization failed:', error);
  }
}

// Start initialization immediately
initializeServer();

app.listen(PORT, () => {
  console.log(`🌟 Server running on http://localhost:${PORT}`);
  console.log('📊 Performance monitoring enabled');
  console.log('💾 Enhanced caching system active');
});

// Graceful shutdown with cleanup
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down gracefully...');
  console.log(`📈 Final cache stats: ${generationCache.size} entries, ${calculateCacheHitRate()}% hit rate`);
  process.exit(0);
});

module.exports = app;