#!/usr/bin/env bash

# Override these values to tune long-running job polling and transient GET retries.
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-10}
POLL_TIMEOUT_SECONDS=${POLL_TIMEOUT_SECONDS:-7200}
HTTP_RETRY_COUNT=${HTTP_RETRY_COUNT:-3}
HTTP_RETRY_DELAY_SECONDS=${HTTP_RETRY_DELAY_SECONDS:-2}
HTTP_CONNECT_TIMEOUT_SECONDS=${HTTP_CONNECT_TIMEOUT_SECONDS:-10}
HTTP_REQUEST_TIMEOUT_SECONDS=${HTTP_REQUEST_TIMEOUT_SECONDS:-60}

curlWithRetry() {
	curl --fail-with-body --silent --show-error \
		--retry "$HTTP_RETRY_COUNT" \
		--retry-all-errors \
		--retry-delay "$HTTP_RETRY_DELAY_SECONDS" \
		--connect-timeout "$HTTP_CONNECT_TIMEOUT_SECONDS" \
		"$@"
}

pollJob() {
	local url_query=$1
	local filter_status=$2
	shift 2

	local response_job
	local status_job
	local status_normalized
	local time_started=$SECONDS

	while ((SECONDS - time_started < POLL_TIMEOUT_SECONDS)); do
		if ! response_job=$(
			curlWithRetry --max-time "$HTTP_REQUEST_TIMEOUT_SECONDS" \
				--request GET --url "$url_query" "$@"
		); then
			printf 'Failed to query async job: %s\n' "$url_query" >&2
			return 1
		fi

		if ! status_job=$(printf '%s\n' "$response_job" | jq -er "$filter_status"); then
			printf 'Async job response has no valid status: %s\n' "$response_job" >&2
			return 1
		fi

		status_normalized=$(printf '%s' "$status_job" | tr '[:upper:]' '[:lower:]')
		printf 'Async job status: %s\n' "$status_job" >&2

		case "$status_normalized" in
			completed | success | succeeded | finished | done)
				printf '%s\n' "$response_job"
				return 0
				;;
			failed | failure | fail | error | cancelled | canceled)
				printf 'Async job failed: %s\n' "$response_job" >&2
				return 1
				;;
			queued | queueing | preparing | processing | running | pending | in_progress | in-progress)
				;;
			*)
				printf 'Unexpected async job status: %s\n' "$status_job" >&2
				return 1
				;;
		esac

		sleep "$POLL_INTERVAL_SECONDS"
	done

	printf 'Async job timed out after %s seconds: %s\n' \
		"$POLL_TIMEOUT_SECONDS" "$url_query" >&2
	return 124
}

pollSglangVideo() {
	local url_base=$1
	local id_video=$2

	pollJob "$url_base/v1/videos/$id_video" '.status'
}

pollMiniMaxTask() {
	local url_base=$1
	local token_api=$2
	local id_task=$3

	pollJob "$url_base/v2/query/video_generation/$id_task" \
		'.task.status // .status' \
		--header "Authorization: Bearer $token_api"
}
