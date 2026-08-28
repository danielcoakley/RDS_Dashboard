import type { NextConfig } from "next";

function originFromUrl(value: string | undefined): string | null {
  if (!value?.trim()) {
    return null;
  }
  try {
    return new URL(value).origin;
  } catch {
    return null;
  }
}

function contentSecurityPolicy(): string {
  const apiOrigin = originFromUrl(process.env.NEXT_PUBLIC_API_BASE_URL);
  const reportUri = apiOrigin ? `${apiOrigin}/security/csp-reports` : null;
  const connectSources = [
    "'self'",
    apiOrigin,
    "https://*.clerk.accounts.dev",
    "https://*.clerk.com",
    "https://*.clerk.dev"
  ].filter(Boolean);

  const directives = [
    "default-src 'self'",
    "base-uri 'self'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "form-action 'self'",
    "img-src 'self' data: blob: https:",
    "font-src 'self' data:",
    "style-src 'self' 'unsafe-inline'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
    `connect-src ${connectSources.join(" ")}`,
    "frame-src https://*.clerk.accounts.dev https://*.clerk.com https://*.clerk.dev",
    "upgrade-insecure-requests"
  ];
  if (reportUri) {
    directives.push(`report-uri ${reportUri}`);
  }
  return directives.join("; ");
}

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy-Report-Only",
            value: contentSecurityPolicy()
          },
          {
            key: "Cross-Origin-Opener-Policy",
            value: "same-origin"
          },
          {
            key: "Cross-Origin-Resource-Policy",
            value: "same-site"
          },
          {
            key: "Permissions-Policy",
            value: "camera=(), microphone=(), geolocation=(), payment=()"
          },
          {
            key: "Referrer-Policy",
            value: "no-referrer"
          },
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains"
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff"
          },
          {
            key: "X-DNS-Prefetch-Control",
            value: "off"
          },
          {
            key: "X-Frame-Options",
            value: "DENY"
          }
        ]
      }
    ];
  }
};

export default nextConfig;
