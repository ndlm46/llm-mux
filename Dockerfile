FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./

RUN go mod download

COPY . .

ARG VERSION=dev
ARG COMMIT=none
ARG BUILD_DATE=unknown

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w -X 'main.Version=${VERSION}' -X 'main.Commit=${COMMIT}' -X 'main.BuildDate=${BUILD_DATE}'" -o ./llm-mux ./cmd/server/

FROM alpine:3.22.0

RUN apk add --no-cache tzdata ca-certificates

RUN mkdir /llm-mux

COPY --from=builder ./app/llm-mux /llm-mux/llm-mux

COPY config.example.yaml /llm-mux/config.example.yaml

WORKDIR /llm-mux

EXPOSE 8318

ENV TZ=UTC

CMD ["./llm-mux"]
