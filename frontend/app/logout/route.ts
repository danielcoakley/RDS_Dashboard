import { NextResponse } from "next/server";
import { clearAppSession } from "../../lib/session";

export async function GET(request: Request) {
  await clearAppSession();
  return NextResponse.redirect(new URL("/login", request.url));
}
