# Requirements Document

## Introduction

This document specifies the requirements for migrating the LLM Council application from OpenRouter API to Poe.com API. The LLM Council is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions through individual responses, anonymized peer review, and chairman synthesis. The migration will replace the OpenRouter backend with the official `fastapi-poe` library while maintaining all existing functionality.

## Glossary

- **LLM Council**: The system that orchestrates multiple LLMs to collaboratively answer questions
- **Council Models**: The set of LLM bots that participate in Stage 1 (individual responses) and Stage 2 (peer rankings)
- **Chairman Model**: The designated LLM bot that synthesizes the final response in Stage 3
- **Poe.com**: A platform that provides access to multiple LLM models through a unified interface
- **fastapi-poe**: The official Poe Python library for querying bots via API key
- **Poe API Key**: A user API key obtained from poe.com/api_key for authenticating requests
- **Bot name**: The display name used by Poe.com to identify models (e.g., "GPT-5", "Claude-Sonnet-4.5", "Gemini-2.5-Pro")
- **Stage 1**: The phase where all council models independently respond to the user query
- **Stage 2**: The phase where models anonymously rank each other's responses
- **Stage 3**: The phase where the chairman synthesizes a final answer
- **ProtocolMessage**: The message format used by fastapi-poe for sending queries
- **PartialResponse**: The streaming response format returned by Poe bots

## Requirements

### Requirement 1

**User Story:** As a developer, I want to configure Poe.com authentication credentials, so that the application can access Poe.com's LLM models.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL load the Poe API key from the POE_API_KEY environment variable
2. WHEN the POE_API_KEY environment variable is missing THEN the system SHALL raise a clear configuration error with instructions to obtain a key from poe.com/api_key
3. THE system SHALL store the Poe API key in a configuration module separate from business logic
4. THE system SHALL never log or expose the API key in error messages or responses

### Requirement 2

**User Story:** As a developer, I want to configure which Poe.com bots serve as council members and chairman, so that I can customize the LLM council composition.

#### Acceptance Criteria

1. THE system SHALL maintain a list of council bot names in the configuration module
2. THE system SHALL maintain a chairman bot name in the configuration module
3. WHEN a bot is configured THEN the system SHALL use Poe.com's display name convention (e.g., "GPT-5", "Claude-Sonnet-4.5", "Gemini-2.5-Pro")
4. THE system SHALL document available bot names in configuration comments for reference

### Requirement 3

**User Story:** As a user, I want the application to query multiple Poe.com bots in parallel, so that Stage 1 responses are collected efficiently.

#### Acceptance Criteria

1. WHEN Stage 1 begins THEN the system SHALL use fastapi-poe's get_bot_response function to query bots
2. WHEN querying council models THEN the system SHALL send messages to all bots concurrently using asyncio.gather
3. WHEN a bot query succeeds THEN the system SHALL accumulate all PartialResponse chunks into a complete response text
4. WHEN a bot query fails THEN the system SHALL log the error and continue with successful responses
5. WHEN all bot queries fail THEN the system SHALL return an appropriate error message

### Requirement 4

**User Story:** As a user, I want the application to stream responses from Poe.com bots, so that I can see partial results as they arrive.

#### Acceptance Criteria

1. WHEN sending a message to a Poe bot THEN the system SHALL use fastapi-poe's async generator to receive PartialResponse chunks
2. WHEN response chunks arrive THEN the system SHALL accumulate the text field into a complete response
3. WHEN streaming completes THEN the system SHALL return the full accumulated response text
4. WHEN a streaming error occurs THEN the system SHALL handle it gracefully and report the failure

### Requirement 5

**User Story:** As a user, I want Stage 2 peer rankings to work with Poe.com bots, so that models can evaluate each other's responses.

#### Acceptance Criteria

1. WHEN Stage 2 begins THEN the system SHALL anonymize Stage 1 responses as "Response A, B, C, etc."
2. WHEN sending ranking prompts THEN the system SHALL query all council bots in parallel via Poe.com
3. WHEN ranking responses arrive THEN the system SHALL parse the "FINAL RANKING:" section
4. THE system SHALL maintain the label-to-model mapping for de-anonymization

### Requirement 6

**User Story:** As a user, I want Stage 3 synthesis to work with a Poe.com chairman bot, so that I receive a final consolidated answer.

#### Acceptance Criteria

1. WHEN Stage 3 begins THEN the system SHALL send the synthesis prompt to the chairman bot via Poe.com
2. WHEN the chairman response arrives THEN the system SHALL return it as the final answer
3. WHEN the chairman query fails THEN the system SHALL return an error message indicating synthesis failure

### Requirement 7

**User Story:** As a developer, I want the Poe.com API calls to be properly managed, so that requests are handled efficiently.

#### Acceptance Criteria

1. WHEN querying a bot THEN the system SHALL pass the API key to each get_bot_response call
2. WHEN multiple requests arrive THEN the system SHALL handle them independently without shared state
3. THE system SHALL respect Poe's rate limit of 500 requests per minute per user
4. WHEN rate limits are approached THEN the system SHALL log a warning

### Requirement 8

**User Story:** As a user, I want conversation title generation to work with Poe.com, so that new conversations get meaningful titles.

#### Acceptance Criteria

1. WHEN a new conversation starts THEN the system SHALL generate a title using a fast Poe.com bot (e.g., "GPT-4o-Mini" or "Claude-3-Haiku")
2. WHEN title generation succeeds THEN the system SHALL update the conversation with the generated title
3. WHEN title generation fails THEN the system SHALL use "New Conversation" as the default title

### Requirement 9

**User Story:** As a developer, I want the existing API endpoints to remain unchanged, so that the frontend continues to work without modification.

#### Acceptance Criteria

1. THE system SHALL maintain the same REST API endpoint structure
2. THE system SHALL return responses in the same JSON format as before
3. THE system SHALL support both streaming and non-streaming message endpoints
4. WHEN SSE events are sent THEN the system SHALL use the same event types and data structure

### Requirement 10

**User Story:** As a developer, I want clear error handling for Poe.com-specific issues, so that problems can be diagnosed and resolved.

#### Acceptance Criteria

1. WHEN authentication fails THEN the system SHALL return a clear error about invalid credentials
2. WHEN a bot is not found THEN the system SHALL return an error identifying the missing bot
3. WHEN rate limits are exceeded THEN the system SHALL return an appropriate error message
4. WHEN the Poe.com service is unavailable THEN the system SHALL return a service unavailable error
