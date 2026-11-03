# static binary, no shell, runs as non-root
FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /llm-gateway ./cmd/llm-gateway

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /llm-gateway /llm-gateway
USER nonroot:nonroot
EXPOSE 8080 9090
ENTRYPOINT ["/llm-gateway"]
CMD ["-config", "/config/gateway.json"]
