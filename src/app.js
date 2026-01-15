import express from "express";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import pinoHttp from "pino-http";
import { webhookRouter } from "./routes/webhook.js";
import { mongoReady } from "./config/mongo.js";

export function createApp(env) {
  const app = express();

  // Logs
  const logger = pinoHttp({
    customSuccessMessage: () => "request completed",
  });
  app.use(logger);

  // Security headers + basic hardening
  app.use(helmet());

  // Rate limit (чтобы не завалили endpoint)
  app.use(
    rateLimit({
      windowMs: 60 * 1000,
      max: 300,
      standardHeaders: true,
      legacyHeaders: false,
    })
  );

  // JSON parser with rawBody for signature verify
  app.use(
    express.json({
      limit: "2mb",
      verify: (req, res, buf) => {
        req.rawBody = buf;
      },
    })
  );

  // Health endpoints
  app.get("/", (req, res) => res.status(200).send("OK"));
  app.get("/healthz", (req, res) => res.status(200).json({ ok: true }));
  app.get("/readyz", (req, res) => {
    const ok = mongoReady();
    res.status(ok ? 200 : 503).json({ mongo: ok });
  });

  // Webhook
  app.use(
    "/webhook",
    webhookRouter({
      verifyToken: env.VERIFY_TOKEN,
      metaAppSecret: env.META_APP_SECRET,
      phoneHashSalt: env.PHONE_HASH_SALT,
      logger: logger.logger,
    })
  );

  return app;
}
