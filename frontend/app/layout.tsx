import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { Plus_Jakarta_Sans } from "next/font/google";
import { isClerkFrontendEnabled } from "../lib/auth-config";
import "./styles.css";

const sans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  display: "swap"
});

export const metadata: Metadata = {
  title: "RDS Energy Analytics",
  description:
    "ISO 50001 energy performance platform for baselines, evidence, and tenant-scoped reporting."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  const content = isClerkFrontendEnabled() ? <ClerkProvider>{children}</ClerkProvider> : children;

  return (
    <html lang="en">
      <body className={sans.className}>{content}</body>
    </html>
  );
}
