FROM golang:1.23-alpine AS build

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/jobs-api .

FROM alpine:3.20

RUN apk add --no-cache ca-certificates && adduser -D -H -u 10001 appuser
USER 10001

WORKDIR /app
COPY --from=build /out/jobs-api /app/jobs-api

EXPOSE 8000
CMD ["/app/jobs-api"]
