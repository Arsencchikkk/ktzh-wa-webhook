export function extractInboundMessages(webhookBody) {
  const out = [];

  const entries = webhookBody?.entry || [];
  for (const entry of entries) {
    const changes = entry?.changes || [];
    for (const change of changes) {
      const value = change?.value || {};
      const metadata = value?.metadata || {};
      const phoneNumberId = metadata?.phone_number_id || null;

      const messages = value?.messages || [];
      for (const m of messages) {
        const waMessageId = m?.id;
        const from = m?.from || null; 
        const ts = m?.timestamp ? new Date(Number(m.timestamp) * 1000) : new Date();
        const type = m?.type || "unknown";

        let text = null;
        if (type === "text") text = m?.text?.body || null;
        else if (type === "button") text = m?.button?.text || null;
        else if (type === "interactive") text = m?.interactive?.type || null; 

        out.push({
          wa_message_id: waMessageId,
          phone_number_id: phoneNumberId,
          sender_phone: from,
          msg_type: type,
          text,
          ts,
          raw_message: m,
        });
      }
    }
  }

  return out;
}
