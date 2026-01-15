import crypto from "crypto";

export function hashPhone(phone, salt) {
  if (!phone) return null;
  const s = salt || "";
  return crypto.createHash("sha256").update(`${s}:${phone}`).digest("hex");
}
