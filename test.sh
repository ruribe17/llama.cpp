curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer no-key' \
--data '{
"messages": [
{
"role": "user",
"content": "Count from 1 to 4097 one at a time, separating each number with a newline. You should not abbreviate the numbers, but list out every single one."
}
],
"n_predict": -2
}'
