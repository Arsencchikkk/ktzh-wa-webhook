import mongoose from "mongoose";

export async function connectMongo(uri) {
  if (!uri) throw new Error("MONGODB_URI is missing");

  mongoose.set("strictQuery", true);

  await mongoose.connect(uri, {
    serverSelectionTimeoutMS: 10000,
  });

  return mongoose.connection;
}

export function mongoReady() {
  return mongoose.connection?.readyState === 1; 
}
