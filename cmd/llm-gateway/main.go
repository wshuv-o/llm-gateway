// Command llm-gateway serves an OpenAI-compatible API in front of several LLM providers.
//
//	llm-gateway -config gateway.json   run the gateway
//	llm-gateway new-key                print a new client key and the sha256 to put in the config
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/wshuv-o/llm-gateway/internal/gateway"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "new-key" {
		b := make([]byte, 24)
		rand.Read(b)
		k := "sk-gw-" + hex.EncodeToString(b)
		sum := sha256.Sum256([]byte(k))
		fmt.Printf("key:    %s   (give this to the client; it is not stored anywhere)\nsha256: %s   (put this in the config)\n",
			k, hex.EncodeToString(sum[:]))
		return
	}
	configPath := flag.String("config", "gateway.json", "path to the config file")
	flag.Parse()
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg, err := gateway.LoadConfig(*configPath)
	if err != nil {
		log.Error("invalid config", "error", err)
		os.Exit(1)
	}
	g := gateway.New(cfg, log)

	api := &http.Server{Addr: cfg.Listen, Handler: g.Handler(), ReadHeaderTimeout: 10 * time.Second}
	servers := []*http.Server{api}
	if cfg.MetricsListen != "" {
		servers = append(servers, &http.Server{Addr: cfg.MetricsListen, Handler: g.MetricsHandler(), ReadHeaderTimeout: 10 * time.Second})
	}
	for _, s := range servers {
		go func() {
			log.Info("listening", "addr", s.Addr)
			if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error("server failed", "addr", s.Addr, "error", err)
				os.Exit(1)
			}
		}()
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	<-ctx.Done()
	// Stop accepting new requests and let in-flight ones (including long streams) finish.
	log.Info("shutting down; draining in-flight requests")
	drain, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	for _, s := range servers {
		if err := s.Shutdown(drain); err != nil {
			log.Warn("drain incomplete", "addr", s.Addr, "error", err)
		}
	}
}
