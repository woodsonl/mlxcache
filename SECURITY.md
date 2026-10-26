# Security policy

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository
(Report a vulnerability in the Security tab). Reports sent that way reach
the maintainer privately; please do not open a public issue for anything
exploitable.

Include what you found, how you reproduced it, and what an attacker gains.
A proof-of-concept is welcome but not required.

## Scope

mlxcache is a loopback service by design: the daemon binds `127.0.0.1`, and
the README's deployment guidance keeps it there. Anything that lets a
non-loopback caller reach the daemon's mutating endpoints, corrupt a
checkpoint that is then served as healthy, or escape the store's byte
budget is in scope. Issues that require the attacker to already control the
machine running the daemon are out of scope.

## Disclosure

I will acknowledge reports within a week and keep reporters informed of the
fix's progress. Once a fix ships, credit goes in the release notes unless
you ask to stay anonymous.
