function toggleForm() {
    const loginContainer = document.getElementById('login-container');
    const signupContainer = document.getElementById('signup-container');
    loginContainer.style.display = (loginContainer.style.display === 'none') ? 'block' : 'none';
    signupContainer.style.display = (signupContainer.style.display === 'none') ? 'block' : 'none';
}

// function sendRequest(url, data, onSuccess, onError) {
//     fetch(url, {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify(data)
//     })
//     .then(response => response.json())
//     .then(data => {
//         if (data.message) {
//             onSuccess(data.message);
//         } else {
//             onError('Server responded without a message.');
//         }
//     })
//     .catch(error => onError('Network error: ' + error.message));
// }

function sendRequest(url, data, onSuccess, onError) {
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url; // Handle redirect
        }
        return response.json();
    })
    .then(data => {
        if (data.message) {
            onSuccess(data.message);
        } else {
            onError('Server responded without a message.');
        }
    })
    .catch(error => onError('Network error: ' + error.message));
}


document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;   
    sendRequest('/login', { username, password }, 
        // message => {
        //     if (message === "Invalid credentials") {
        //         alert(message);
        //     }
        // },
        error => alert(error));
});

document.getElementById('signup-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('new-username').value;
    const password = document.getElementById('new-password').value;
    sendRequest('/signup', { username, password }, 
        message => {
            alert(message);
            toggleForm(); // Switch back to login form after successful signup
        },
        error => alert(error));
});