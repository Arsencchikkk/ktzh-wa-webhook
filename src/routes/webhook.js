import express from "express";
import { verifyMetaSignature } from "../services/metaSignature.js";
import { extractInboundMessages } from "../services/whatsappParser.js";
import { Message } from "../models/Message.js";
import { hashPhone } from "../services/privacy.js";

export function webhookRouter({ verifyToken, metaAppSecret, phoneHashSalt, logger }) {
  const router = express.Router();


  router.get("/", (req, res) => {
    const mode = req.query["hub.mode"];
    const token = req.query["hub.verify_token"];
    const challenge = req.query["hub.challenge"];

    if (mode === "subscribe" && token === verifyToken) {
      return res.status(200).send(challenge);
    }
    return res.sendStatus(403);
  });

  router.post("/", async (req, res) => {
  
    if (!verifyMetaSignature(req, metaAppSecret)) {
      return res.sendStatus(403);
    }

    res.sendStatus(200);


    try {
      const msgs = extractInboundMessages(req.body);

      if (!msgs.length) {
        logger.info({ kind: "webhook", note: "no messages" });
        return;
      }

      const ops = msgs
        .filter((m) => m.wa_message_id)
        .map((m) => ({
          updateOne: {
            filter: { wa_message_id: m.wa_message_id },
            update: {
              $setOnInsert: {
                wa_message_id: m.wa_message_id,
                phone_number_id: m.phone_number_id,
                sender_phone_hash: hashPhone(m.sender_phone, phoneHashSalt),
                direction: "inbound",
                msg_type: m.msg_type,
                text: m.text,
                ts: m.ts,
                raw_message: m.raw_message,
              },
            },
            upsert: true,
          },
        }));

      if (ops.length) {
        await Message.bulkWrite(ops, { ordered: false });
      }

      logger.info({ kind: "webhook", saved: ops.length });
    } catch (e) {
      logger.error({ kind: "webhook_error", err: e?.message || String(e) });
    }
  });

  return router;
}
