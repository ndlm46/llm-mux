// Package misc provides miscellaneous utility functions and embedded data for the CLI Proxy API.
// This package contains general-purpose helpers and embedded resources that do not fit into
// more specific domain packages. It includes embedded instructional text for Codex-related operations.
package misc

import (
	"embed"
	"strings"
)

//go:embed codex_instructions
var codexInstructionsDir embed.FS

func CodexInstructionsForModel(modelName, systemInstructions string) (bool, string) {
	entries, _ := codexInstructionsDir.ReadDir("codex_instructions")

	lastPrompt := ""
	lastCodexPrompt := ""
	lastCodexMaxPrompt := ""
	last51Prompt := ""
	for _, entry := range entries {
		content, _ := codexInstructionsDir.ReadFile("codex_instructions/" + entry.Name())
		if strings.HasPrefix(systemInstructions, string(content)) {
			return true, ""
		}
		name := entry.Name()
		if strings.HasPrefix(name, "gpt_5_codex_prompt.md") {
			lastCodexPrompt = string(content)
		} else if strings.HasPrefix(name, "gpt-5.1-codex-max_prompt.md") {
			lastCodexMaxPrompt = string(content)
		} else if strings.HasPrefix(name, "prompt.md") {
			lastPrompt = string(content)
		} else if strings.HasPrefix(name, "gpt_5_1_prompt.md") {
			last51Prompt = string(content)
		}
	}
	if strings.Contains(modelName, "codex-max") {
		return false, lastCodexMaxPrompt
	} else if strings.Contains(modelName, "codex") {
		return false, lastCodexPrompt
	} else if strings.Contains(modelName, "5.1") {
		return false, last51Prompt
	} else {
		return false, lastPrompt
	}
}
