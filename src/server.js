import { createApp } from "./app.js";
import { connectMongo } from "./config/mongo.js";

const env = {
  PORT: process.env.PORT || 3000,
  MONGODB_URI: process.env.MONGODB_URI,
  VERIFY_TOKEN: process.env.VERIFY_TOKEN,
  META_APP_SECRET: process.env.META_APP_SECRET || "",
  PHONE_HASH_SALT: process.env.PHONE_HASH_SALT || "",
};

if (!env.VERIFY_TOKEN) {
  console.error("VERIFY_TOKEN is missing");
  process.exit(1);
}
if (!env.MONGODB_URI) {
  console.error("MONGODB_URI is missing");
  process.exit(1);
}

await connectMongo(env.MONGODB_URI);

const app = createApp(env);
app.listen(env.PORT, () => {
  console.log(`Server listening on port ${env.PORT}`);
});
