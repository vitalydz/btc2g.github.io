(function(){
  const endpoint='https://formspree.io/f/mkoyjwkr';
  document.querySelectorAll('.article-signup-form').forEach(form=>form.addEventListener('submit',async event=>{
    event.preventDefault();
    const input=form.querySelector('input[type=email]');const button=form.querySelector('button');const status=form.parentElement.querySelector('.article-signup-status');const email=input.value.trim();
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){status.textContent='Enter a valid email address.';input.focus();return;}
    button.disabled=true;status.textContent='Subscribing…';
    try{const response=await window.BTC2GSubscription.send(endpoint,{method:'POST',headers:{Accept:'application/json','Content-Type':'application/json'},body:JSON.stringify({email,source:form.dataset.source})});if(!response.ok)throw new Error('Subscription failed');form.reset();status.textContent='Request received. Check your inbox for future BTC2G updates.';}catch(error){status.textContent=error.name==='TimeoutError'?'The result is uncertain. Check your inbox before retrying.':'Subscription failed. Please try again later.';}finally{button.disabled=false;}
  }));
})();
