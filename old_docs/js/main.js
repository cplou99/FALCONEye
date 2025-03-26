// add a copy button to code blocks
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('pre').forEach(function(pre) {
        // create container around the pre block
        const container = document.createElement('div');
        container.className = 'code-container';

        // insert the button
        const button = document.createElement('button');
        button.innerHTML = '<i class="bi bi-copy"></i>';
        button.className = 'clipboard-button';
        container.appendChild(button);

        // insert the pre block
        pre.parentNode.insertBefore(container, pre);
        container.appendChild(pre);

        button.addEventListener('click', function() {
            const codeText = pre.innerText;
            // use the clipboard api to copy the text
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(codeText).then(() => {
                    button.innerHTML = '<i class="bi bi-check-lg"></i>';
                    setTimeout(() => {
                        button.innerHTML = '<i class="bi bi-copy"></i>';
                    }, 2000);
                }).catch(err => {
                    console.error('Clipboard error:', err);
                });
            } else {
                console.log('Clipboard API is not available');
            }
        });
    });
});
