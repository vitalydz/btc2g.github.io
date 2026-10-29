import { request } from './metrics-view.js';

const button = document.querySelector('#confirm-email');
const message = document.querySelector('#verify-message');
const token = new URLSearchParams(location.search).get('token');

if (!token) {
  button.disabled = true;
  message.textContent = 'Invalid verification link.';
}

button.addEventListener('click', async () => {
  button.disabled = true;
  message.textContent = '';
  message.classList.remove('success');
  try {
    await request('/api/email-verification/confirm', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ token })
    });
    message.textContent = 'Email verified. You can return to your account.';
    message.classList.add('success');
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
  }
});
