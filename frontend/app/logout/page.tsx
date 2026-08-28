import { redirect } from "next/navigation";
import { clearAppSession } from "../../lib/session";

export default async function LogoutPage() {
  await clearAppSession();
  redirect("/login");
}
