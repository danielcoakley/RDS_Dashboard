import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#effdf5",
          100: "#d9fae9",
          200: "#b4f4d3",
          300: "#7ee8b5",
          400: "#40d18f",
          500: "#1ab672",
          600: "#0f9359",
          700: "#0f7549",
          800: "#115c3c",
          900: "#114c33",
          950: "#03291e",
        },
        ink: {
          50: "#f6f7f9",
          100: "#ebeef3",
          200: "#d4dae3",
          300: "#aeb9ca",
          400: "#8295b0",
          500: "#647696",
          600: "#505f7d",
          700: "#414d66",
          800: "#384156",
          900: "#313848",
          950: "#1f2433",
        },
      },
    },
  },
  plugins: [],
};
export default config;
