chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action == "fetchContent") {
        console.log("fteching");
        fetch('http://127.0.0.1:1234/result/', {
            method: 'POST',
            body: JSON.stringify({ par: request.content }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            // Send the entities back to the content script
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                const activeTab = tabs[0];
                console.log("sendback highlight");
                chrome.tabs.sendMessage(activeTab.id, {action: "highlightEntities", entities: data});
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });

        return true;  // Will respond asynchronously.
    }
});