import mongoose from "mongoose";

const MessageSchema = new mongoose.Schema(
  {
    wa_message_id: { type: String, unique: true, index: true },
    phone_number_id: { type: String, index: true },
    sender_phone_hash: { type: String, index: true },
    direction: { type: String, enum: ["inbound", "outbound"], index: true },
    msg_type: { type: String, index: true },
    text: { type: String, default: null },
    ts: { type: Date, index: true },
    raw_message: { type: mongoose.Schema.Types.Mixed, default: null }
  },
  {
    timestamps: true,
    collection: "message"   
  }
);

export const Message = mongoose.model("Message", MessageSchema);
