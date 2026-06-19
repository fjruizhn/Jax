#!/bin/bash
ls -t ~/jax/missions/web-task-*_result.md 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
ls -t ~/jax/missions/web-task-*.md 2>/dev/null | grep -v "_result" | tail -n +11 | xargs rm -f 2>/dev/null
find ~/jax -name "*.backup*" -mtime +30 -delete 2>/dev/null
find ~/jax-platform -name "*.backup*" -mtime +30 -delete 2>/dev/null
echo "[$(date)] cleanup completado"
