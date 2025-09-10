document.onkeydown = checkKey;
function checkKey(e) {
    // This will prevent the use spx-gc.js checkKey() -function
    // and does not interfere with spaces, for example.
    return;
} // checkKey

function e(ID) {
    return document.getElementById(ID);
} // e

window.addEventListener("beforeunload", function (event) {
    sendReturnValue();
});

function sendReturnValue(closeWindow = false) {
    const selNro = document.querySelector('input[name="imageSelect"]:checked').value;
    const url = e('img' + selNro).src;
    console.log(`Selected image: ${url}`);

    if (window.opener) {
        const urlParams = new URLSearchParams(window.location.search);
        const id = urlParams.get('id');
        const index = urlParams.get('index');
        let returnData = {
            epochID: id,
            fieldIX: index,
            infoTxt: `Image: ${selNro}`,
            value: url

        };
        window.opener.handleReturnValueFromPlugin(returnData);
    }

    if (closeWindow) {
        window.close();
    }
}



window.addEventListener('load', async () => {
    console.clear()
    try {
        e('playBtn').addEventListener('click', () => { sendReturnValue(true) });
    } catch (error) {
        console.error('HOUSTON!', error)
    }
}); // load event listener

window.addEventListener("beforeunload", function (event) {
    sendReturnValue();
});


