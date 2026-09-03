# =============================================================================
# PhishGuard AI — frontend image
#
# Two stages: Node builds the static bundle, nginx serves it. The runtime image
# contains no Node and no source, only the compiled assets.
#
#   docker build -f frontend/Dockerfile -t phishguard-web ./frontend
# =============================================================================

# --- build -------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Vite inlines VITE_* variables at build time, so the API URL must be supplied
# here rather than at container start. It is public by design — never pass a
# secret through this mechanism.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# --- runtime -----------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
