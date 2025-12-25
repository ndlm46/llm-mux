---
name: llm-mux-test
description: Test llm-mux IR translator - cross-format API translation
---

## Quick Check

```bash
PORT=8318; curl -s http://localhost:$PORT/v1/models | jq -r '.data[:3][].id'
```

## Translation Matrix Tests

Test các luồng translation thực tế trong hệ thống.

### OpenAI Format -> Gemini Backend

```bash
PORT=8318
# Chat
curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}]}' | jq -r '.choices[0].message.content'

# Tool Call
curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Weather Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}]}' | jq '.choices[0].message.tool_calls[0].function'
```

### Claude Format -> Gemini Backend

```bash
PORT=8318
# Note: Response may have thinking block first, so find text block
curl -s http://localhost:$PORT/v1/messages -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" \
  -d '{"model":"gemini-2.5-flash","max_tokens":50,"messages":[{"role":"user","content":"Hi"}]}' | jq -r '.content[] | select(.type=="text") | .text'
```

### Gemini Format -> Codex/GPT Backend

```bash
PORT=8318
curl -s "http://localhost:$PORT/v1beta/models/gpt-5:generateContent" -H "Content-Type: application/json" \
  -d '{"contents":[{"role":"user","parts":[{"text":"Hi"}]}]}' | jq -r '.candidates[0].content.parts[0].text'
```

### OpenAI Format -> Claude Backend

```bash
PORT=8318
curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hi"}]}' | jq -r '.choices[0].message.content'
```

## Streaming Test

```bash
PORT=8318
curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"Hi"}],"stream":true}' | head -3
```

## Thinking/Reasoning Test

```bash
PORT=8318
curl -s http://localhost:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"2+2=?"}],"reasoning_effort":"low"}' | jq '.choices[0].message | {content, reasoning: .reasoning_content[:100]}'
```
