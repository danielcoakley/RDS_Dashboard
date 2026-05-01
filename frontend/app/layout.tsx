import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
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
  return (
    <html lang="en">
      <body className={sans.className}>{children}</body>
    </html>
  );
}
