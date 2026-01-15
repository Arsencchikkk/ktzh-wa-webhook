import crypto from "crypto";

export function verifyMetaSignature(req, appSecret) {
  if (!appSecret) return true;

  const sig = req.get("x-hub-signature-256");
  if (!sig) return false;

  const expected =
    "sha256=" +
    crypto
      .createHmac("sha256", appSecret)
      .update(req.rawBody || Buffer.from(""))
      .digest("hex");

  try {
    return crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(expected));
  } catch {
    return false;
  }
}
