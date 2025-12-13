package executor

// altContextKey is an unexported context key type for the "alt" value.
// Using a struct type prevents collisions with string keys from other packages.
type altContextKey struct{}
