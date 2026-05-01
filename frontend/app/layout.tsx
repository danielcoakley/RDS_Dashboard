import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "RDS Energy Analytics",
  description: "ISO 50001 energy analytics SaaS platform"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
