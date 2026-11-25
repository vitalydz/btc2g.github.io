(() => {
  const rows = {
    en:['Also send me the Sunday research digest in English.','Save digest preference','An email will ask you to confirm this preference.','Request received. Confirm the preference using the link in your email.','Could not confirm receipt. Your preference may not have changed. No automatic retry was sent.','Sending…'],
    ru:['Также присылать воскресный исследовательский дайджест на английском.','Сохранить выбор дайджеста','Настройка применяется после подтверждения по ссылке в письме.','Запрос получен. Подтвердите настройку по ссылке в письме.','Не удалось подтвердить получение запроса. Настройка могла не измениться. Автоматического повтора не было.','Отправка…'],
    he:['שלחו גם תקציר מחקר באנגלית בימי ראשון.','שמירת העדפת תקציר','ההעדפה תחול לאחר אישור בקישור בדואר.','הבקשה התקבלה. אשרו את ההעדפה בקישור בדואר.','לא ניתן לאשר את קבלת הבקשה. ייתכן שההעדפה לא השתנתה. לא בוצע ניסיון נוסף אוטומטי.','שולח…'],
    de:['Auch den Forschungsdigest am Sonntag auf Englisch senden.','Digest-Auswahl speichern','Die Einstellung gilt erst nach Bestätigung per E-Mail.','Anfrage eingegangen. Bitte per E-Mail-Link bestätigen.','Eingang nicht bestätigt. Die Einstellung wurde möglicherweise nicht geändert. Kein automatischer Neuversand.','Wird gesendet…'],
    fr:['Recevoir aussi le résumé de recherche du dimanche en anglais.','Enregistrer le choix','Le choix sera appliqué après confirmation par e-mail.','Demande reçue. Confirmez via le lien envoyé par e-mail.','Réception non confirmée. Le choix peut être inchangé. Aucun nouvel envoi automatique.','Envoi…'],
    es:['Recibir también el resumen de investigación del domingo en inglés.','Guardar elección','La preferencia se aplica tras confirmar por correo.','Solicitud recibida. Confirma con el enlace del correo.','Recepción no confirmada. Puede que la preferencia no haya cambiado. No se reenvió automáticamente.','Enviando…'],
    zh:['同时接收周日英文研究摘要。','保存摘要选择','通过邮件链接确认后，设置才会生效。','已收到请求。请通过邮件链接确认设置。','无法确认是否收到请求。设置可能未更改。没有自动重试。','正在发送…'],
    it:['Invia anche il riepilogo di ricerca domenicale in inglese.','Salva scelta','La preferenza si applica dopo la conferma via email.','Richiesta ricevuta. Conferma tramite il link email.','Ricezione non confermata. La preferenza potrebbe non essere cambiata. Nessun reinvio automatico.','Invio…'],
    pt:['Receber também o resumo de pesquisa de domingo em inglês.','Guardar escolha','A preferência aplica-se após confirmação por email.','Pedido recebido. Confirme pelo link no email.','Receção não confirmada. A preferência pode não ter mudado. Sem reenvio automático.','A enviar…'],
    ar:['إرسال ملخص البحث يوم الأحد باللغة الإنجليزية أيضًا.','حفظ اختيار الملخص','يطبق الاختيار بعد التأكيد عبر رابط البريد.','تم استلام الطلب. أكد الاختيار عبر رابط البريد.','تعذر تأكيد الاستلام. ربما لم يتغير الاختيار. لم تتم إعادة المحاولة تلقائيًا.','جارٍ الإرسال…'],
    hi:['रविवार का अंग्रेज़ी शोध सारांश भी भेजें।','सारांश विकल्प सहेजें','ईमेल लिंक से पुष्टि के बाद विकल्प लागू होगा।','अनुरोध मिला। ईमेल लिंक से विकल्प की पुष्टि करें।','प्राप्ति की पुष्टि नहीं हुई। विकल्प शायद नहीं बदला। अपने आप दोबारा नहीं भेजा गया।','भेज रहे हैं…'],
    ja:['日曜日の英語の研究ダイジェストも受け取る。','設定を保存','メールのリンクで確認後に設定が適用されます。','リクエストを受信しました。メールのリンクで確認してください。','受信を確認できません。設定は変更されていない可能性があります。自動再送はしていません。','送信中…'],
    ko:['일요일 영어 리서치 요약도 받겠습니다.','설정 저장','이메일 링크로 확인한 후 설정이 적용됩니다.','요청을 받았습니다. 이메일 링크로 설정을 확인하세요.','수신을 확인할 수 없습니다. 설정이 바뀌지 않았을 수 있습니다. 자동 재전송하지 않았습니다.','전송 중…'],
    tr:['Pazar günleri İngilizce araştırma özetini de gönderin.','Seçimi kaydet','Tercih, e-posta bağlantısıyla onaylandıktan sonra uygulanır.','Talep alındı. E-postadaki bağlantıyla onaylayın.','Alındığı doğrulanamadı. Tercih değişmemiş olabilir. Otomatik tekrar gönderilmedi.','Gönderiliyor…'],
    pl:['Wysyłaj także niedzielne podsumowanie badań po angielsku.','Zapisz wybór','Wybór zacznie obowiązywać po potwierdzeniu e-mailem.','Zgłoszenie otrzymano. Potwierdź wybór linkiem w e-mailu.','Nie potwierdzono odbioru. Wybór mógł się nie zmienić. Bez automatycznej ponownej wysyłki.','Wysyłanie…'],
    nl:['Stuur ook de Engelse onderzoeksdigest op zondag.','Keuze opslaan','De keuze geldt na bevestiging via de e-maillink.','Aanvraag ontvangen. Bevestig via de e-maillink.','Ontvangst niet bevestigd. De keuze is mogelijk niet gewijzigd. Niet automatisch opnieuw verzonden.','Verzenden…'],
    uk:['Також надсилати недільний дослідницький дайджест англійською.','Зберегти вибір','Налаштування застосовується після підтвердження посиланням у листі.','Запит отримано. Підтвердьте налаштування посиланням у листі.','Отримання не підтверджено. Налаштування могло не змінитися. Автоматичного повтору не було.','Надсилання…'],
    id:['Kirim juga ringkasan riset hari Minggu dalam bahasa Inggris.','Simpan pilihan','Pilihan diterapkan setelah konfirmasi melalui tautan email.','Permintaan diterima. Konfirmasikan melalui tautan email.','Penerimaan belum dikonfirmasi. Pilihan mungkin belum berubah. Tidak dikirim ulang otomatis.','Mengirim…'],
    vi:['Gửi thêm bản tin nghiên cứu Chủ nhật bằng tiếng Anh.','Lưu lựa chọn','Lựa chọn có hiệu lực sau khi xác nhận qua liên kết email.','Đã nhận yêu cầu. Hãy xác nhận qua liên kết email.','Chưa xác nhận đã nhận yêu cầu. Lựa chọn có thể chưa thay đổi. Không tự động gửi lại.','Đang gửi…']
  };
  let email='',lang='en',busy=false;
  const node=id=>document.getElementById(id);
  const text=()=>rows[lang];
  function setLanguage(value){
    lang=Object.hasOwn(rows,value)?value:'en';
    node('weekly-digest-choice-label').textContent=text()[0];
    node('save-digest-preferences').textContent=text()[busy?5:1];
    node('digest-preferences-help').textContent=text()[2];
  }
  async function submit(event){
    event.preventDefault();if(busy)return;
    const checkbox=node('weekly-digest-choice'),button=node('save-digest-preferences'),status=node('digest-preferences-status');
    const weeklyDigest=checkbox.checked;
    busy=true;checkbox.disabled=true;button.disabled=true;status.textContent='';button.textContent=text()[5];
    try{
      if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))throw Error('Email required');
      const configResponse=await fetch('assets/audience-config.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});
      if(!configResponse.ok)throw Error('Configuration unavailable');
      const config=await configResponse.json();
      if(typeof config.apiOrigin!=='string')throw Error('Service not connected');
      const base=new URL(config.apiOrigin);
      if(base.protocol!=='https:'||base.username||base.password||base.pathname!=='/'||base.search||base.hash)throw Error('Invalid service origin');
      const response=await fetch(new URL('/api/digest-request',base).href,{method:'POST',credentials:'omit',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,weeklyDigest,consent:true,consentVersion:'2026-10-29',lang}),signal:AbortSignal.timeout(15000)});
      if(!response.ok)throw Error('Receipt unavailable');
      const receipt=await response.json();
      if(receipt.status!=='request_received'||receipt.confirmed!==false)throw Error('Unexpected receipt');
      status.textContent=text()[3];
    }catch{status.textContent=text()[4];}
    finally{busy=false;checkbox.disabled=false;button.disabled=false;button.textContent=text()[1];}
  }
  node('digest-preferences-form').addEventListener('submit',submit);
  window.BTC2GDigest={setLanguage,setEmail(value){email=String(value||'');node('digest-preferences-status').textContent='';},translations:rows};
  setLanguage('en');
})();
