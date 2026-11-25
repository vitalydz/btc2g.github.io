(function(){
  document.querySelectorAll('.article-signup-form').forEach(form=>form.addEventListener('submit',async event=>{
    event.preventDefault();
    const input=form.querySelector('input[type=email]');
    const consent=form.querySelector('input[name=consent]');
    const button=form.querySelector('button');
    const status=form.parentElement.querySelector('.article-signup-status');
    const email=input.value.trim();
    if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){status.textContent='Enter a valid email address.';input.focus();return;}
    if(!consent?.checked){status.textContent='Please agree to receive BTC2G product updates.';consent?.focus();return;}
    button.disabled=true;status.textContent='Sending request…';
    try{
      await window.BTC2GSubscription.subscribe(email,true,'en');
      form.reset();status.textContent='Request received. Check your email and confirm using the link. Your subscription starts only after confirmation. This does not create a BTC2G account.';
    }catch(error){status.textContent=window.BTC2GSubscription.text('en').uncertain;}
    finally{button.disabled=false;}
  }));
})();
