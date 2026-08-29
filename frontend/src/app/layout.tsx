import type { Metadata } from 'next';
import './globals.css';
import { AuthProvider } from '@/lib/auth';

export const metadata: Metadata = {
  title: 'EnMS — ISO 50001 Energy Management Platform',
  description: 'Achieve ISO 50001 certification with automated energy baseline modeling, weather-normalized analytics, and full EnMS workflow.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
