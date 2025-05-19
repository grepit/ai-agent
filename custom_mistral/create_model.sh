#!/bin/bash
CURRENT_DATE=$(date "+%A, %B %d, %Y")
CURRENT_TIME=$(date "+%H:%M:%S")
cat > Modelfile << EOL
FROM mistral
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM """
Today is $CURRENT_DATE. The current time is $CURRENT_TIME. The current year is $(date +%Y). Always use this information when responding to time-related queries.
"""
EOL

ollama create mistral-datetime -f Modelfile
echo "Created model with date: $CURRENT_DATE and time: $CURRENT_TIME"
