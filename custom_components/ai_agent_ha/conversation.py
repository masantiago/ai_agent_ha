"""Conversation support for AI Agent HA."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation agent from a config entry."""
    agent = AIAgentConversation(hass, config_entry)
    async_add_entities([agent])


class AIAgentConversation(conversation.ConversationEntity):
    """AI Agent HA conversation agent wrapper."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self.config_entry = config_entry
        self._attr_name = f"AI Agent ({config_entry.data.get('ai_provider', 'Unknown')})"
        self._attr_unique_id = f"{config_entry.entry_id}_conversation"
        
        # Get reference to the existing AI agent data
        self._integration_data = hass.data[DOMAIN][config_entry.entry_id]
        self._last_automation_suggestion = None  # Store last automation for confirmation

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "zh", "ja", "ko"]

    @property
    def attribution(self) -> str:
        """Return attribution."""
        provider = self.config_entry.data.get('ai_provider', 'Unknown')
        return f"Powered by {provider.title()} via AI Agent HA"

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a conversation input."""
        try:
            # Check if user is confirming an automation creation
            user_text = user_input.text.lower().strip()
            confirmation_phrases = [
                "yes", "sí", "si", "approve", "create it", "créala", "create the automation", 
                "crea la automatización", "confirm", "confirma", "ok", "vale", "adelante"
            ]
            
            is_confirmation = any(phrase in user_text for phrase in confirmation_phrases)
            is_short = len(user_text.split()) <= 3  # Short responses are likely confirmations
            
            # If user is confirming and we have a stored automation, try to create it
            if is_confirmation and is_short and self._last_automation_suggestion:
                _LOGGER.warning("DEBUG CONV: User confirming automation, attempting to create")
                try:
                    create_result = await self.hass.services.async_call(
                        DOMAIN,
                        "create_automation",
                        {"automation": self._last_automation_suggestion},
                        blocking=True,
                        return_response=True
                    )
                    
                    _LOGGER.warning("DEBUG CONV: Create automation result: %s", create_result)
                    
                    if create_result and create_result.get("success"):
                        response = "Automation created successfully!"
                    else:
                        error_msg = create_result.get("error", "Unknown error") if create_result else "No response"
                        response = f"Error creating automation: {error_msg}"
                    
                    # Clear the stored automation
                    self._last_automation_suggestion = None
                    
                except Exception as create_error:
                    _LOGGER.error("Error creating automation: %s", create_error)
                    response = f"Error creating automation: {create_error}"
            else:
                # Normal processing
                response = await self._process_with_ai_agent(user_input.text)
            
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(response)
            
            return conversation.ConversationResult(
                response=intent_response, 
                conversation_id=user_input.conversation_id
            )
            
        except Exception as err:
            _LOGGER.error("Error processing conversation: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I encountered an error: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, 
                conversation_id=user_input.conversation_id
            )

    async def _process_with_ai_agent(self, message: str) -> str:
        """Process message using the existing AI Agent HA service."""
        try:
            # Call the existing ai_agent_ha.query service
            result = await self.hass.services.async_call(
                DOMAIN,
                "query",
                {"prompt": message},
                blocking=True,
                return_response=True
            )
            
            # Debug what we received
            _LOGGER.warning("DEBUG CONV: Received result type: %s", type(result))
            _LOGGER.warning("DEBUG CONV: Received result keys: %s", list(result.keys()) if isinstance(result, dict) else "not dict")
            if isinstance(result, dict) and "request_type" in result:
                _LOGGER.warning("DEBUG CONV: Found request_type: %s", result["request_type"])
            
            # Extract response from service result - handle the actual ai_agent_ha structure
            if isinstance(result, dict):
                # Check for error first
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Handle different request types FIRST (most specific)
                if "request_type" in result:
                    request_type = result["request_type"]
                    _LOGGER.warning("DEBUG CONV: Processing request_type: %s", request_type)
                    
                    if request_type == "final_response":
                        # Handle final responses - this is the clean answer
                        response_content = result.get("response", result.get("message", "Response received"))
                        _LOGGER.warning("DEBUG CONV: Final response content type: %s", type(response_content))
                        
                        # Check if the final response is a list (like automations list)
                        if isinstance(response_content, list):
                            _LOGGER.warning("DEBUG CONV: Final response is a list, formatting...")
                            return self._format_list_response(response_content)
                        else:
                            _LOGGER.warning("DEBUG CONV: Returning final_response as string: %s", response_content)
                            return str(response_content)
                    
                    elif request_type == "automation_suggestion":
                        # For automation suggestions, format nicely and store for potential creation
                        _LOGGER.warning("DEBUG CONV: Processing automation_suggestion")
                        message_text = result.get("message", "I've created an automation suggestion.")
                        
                        if "automation" in result:
                            automation = result["automation"]
                            # Store for potential confirmation
                            self._last_automation_suggestion = automation
                            
                            alias = automation.get("alias", "Unknown automation")
                            description = automation.get("description", "")
                            
                            # Create a more natural response
                            response = f"{message_text}\n\n**{alias}**"
                            if description:
                                response += f"\n{description}"
                            
                            # Add trigger info if available
                            if "trigger" in automation and automation["trigger"]:
                                trigger = automation["trigger"][0]  # Get first trigger
                                if trigger.get("platform") == "time":
                                    response += f"\n\nTrigger: Daily at {trigger.get('at', 'unknown time')}"
                                elif trigger.get("platform") == "state":
                                    entity = trigger.get("entity_id", "entity")
                                    response += f"\nTrigger: When {entity} changes"
                                elif trigger.get("platform") == "device":
                                    response += f"\nTrigger: Device trigger"
                            
                            # Add action info
                            if "action" in automation and automation["action"]:
                                action = automation["action"][0]  # Get first action
                                if "service" in action:
                                    service = action["service"]
                                    target = action.get("target", {}).get("entity_id", "entity")
                                    response += f"\nAction: {service} on {target}"
                            
                            response += "\n\nSay 'yes' or 'create it' to create this automation, or 'no' to cancel."
                            
                            _LOGGER.warning("DEBUG CONV: Returning formatted automation suggestion")
                            return response
                        else:
                            return message_text
                    
                    elif request_type == "data_request":
                        # Handle data requests
                        return result.get("message", result.get("answer", "Data request processed"))
                    
                    elif request_type == "action_request":
                        # Handle action requests
                        return result.get("message", result.get("answer", "Action completed"))
                    
                    else:
                        # Unknown request type, try to get message or answer
                        return result.get("message", result.get("answer", f"Processed {request_type}"))
                
                # ai_agent_ha standard response structure (fallback)
                elif "answer" in result:
                    answer = result["answer"]
                    _LOGGER.warning("DEBUG CONV: Found answer field, type: %s", type(answer))
                    
                    # Check if answer is a JSON string that needs parsing
                    if isinstance(answer, str) and answer.strip().startswith("{"):
                        _LOGGER.warning("DEBUG CONV: Answer looks like JSON string, attempting to parse...")
                        try:
                            import json
                            parsed_answer = json.loads(answer)
                            _LOGGER.warning("DEBUG CONV: Successfully parsed JSON, type: %s", type(parsed_answer))
                            
                            # Now handle the parsed object
                            if isinstance(parsed_answer, dict):
                                if parsed_answer.get("request_type") == "automation_suggestion":
                                    # Store automation for potential creation
                                    if "automation" in parsed_answer:
                                        self._last_automation_suggestion = parsed_answer["automation"]
                                    return self._format_automation_suggestion(parsed_answer)
                                else:
                                    return self._format_dict_response(parsed_answer)
                            elif isinstance(parsed_answer, list):
                                return self._format_list_response(parsed_answer)
                            else:
                                return str(parsed_answer)
                                
                        except json.JSONDecodeError:
                            _LOGGER.warning("DEBUG CONV: Failed to parse as JSON, treating as string")
                            return str(answer)
                    
                    # Check if answer is already a list (like automations list)
                    elif isinstance(answer, list):
                        _LOGGER.warning("DEBUG CONV: Answer is a list, formatting...")
                        return self._format_list_response(answer)
                    # Check if answer is a dict (like single entity info)
                    elif isinstance(answer, dict):
                        _LOGGER.warning("DEBUG CONV: Answer is a dict, formatting...")
                        return self._format_single_entity_response(answer)
                    else:
                        _LOGGER.warning("DEBUG CONV: Answer is string/other: %s", answer)
                        return str(answer)
                
                # Check if it's a success response with answer
                elif result.get("success") and "answer" in result:
                    answer = result["answer"]
                    _LOGGER.warning("DEBUG CONV: Found success+answer, type: %s", type(answer))
                    
                    # Handle JSON string in success response too
                    if isinstance(answer, str) and answer.strip().startswith("{"):
                        try:
                            import json
                            parsed_answer = json.loads(answer)
                            if isinstance(parsed_answer, dict) and parsed_answer.get("request_type") == "automation_suggestion":
                                if "automation" in parsed_answer:
                                    self._last_automation_suggestion = parsed_answer["automation"]
                                return self._format_automation_suggestion(parsed_answer)
                            else:
                                return self._format_dict_response(parsed_answer)
                        except json.JSONDecodeError:
                            return str(answer)
                    elif isinstance(answer, list):
                        return self._format_list_response(answer)
                    elif isinstance(answer, dict):
                        return self._format_single_entity_response(answer)
                    else:
                        return str(answer)
                
                # Look for other common response fields
                else:
                    response_fields = [
                        "response", "text", "message", "content", "result", "output", 
                        "reply", "final_response", "ai_response"
                    ]
                    
                    for key in response_fields:
                        if key in result and result[key]:
                            response_value = result[key]
                            
                            # If the response is also a dict, try to extract text from it
                            if isinstance(response_value, dict):
                                for subkey in ["text", "content", "message", "response", "answer"]:
                                    if subkey in response_value:
                                        return str(response_value[subkey])
                                # If dict but no text fields, convert to readable string
                                return self._format_dict_response(response_value)
                            
                            return str(response_value)
                    
                    # If no standard response fields found, format as readable text
                    _LOGGER.warning("DEBUG CONV: No standard response fields found. Available keys: %s", list(result.keys()))
                    return self._format_dict_response(result)
                
            elif isinstance(result, list):
                # Handle direct list responses
                return self._format_list_response(result)
                
            elif isinstance(result, str) and result.strip():
                return result
            elif result is not None:
                return str(result)
            else:
                # Try direct access as fallback
                return await self._direct_ai_call(message)
                
        except Exception as service_error:
            _LOGGER.error("Error calling ai_agent_ha.query service: %s", service_error)
            
            # Fallback: try to access the AI client directly if service fails
            try:
                return await self._direct_ai_call(message)
            except Exception as direct_error:
                _LOGGER.error("Error with direct AI call: %s", direct_error)
                raise HomeAssistantError(f"Unable to process request: {service_error}")

    def _format_automation_suggestion(self, suggestion_data: dict) -> str:
        """Format an automation suggestion into readable text."""
        try:
            message_text = suggestion_data.get("message", "I've created an automation suggestion.")
            
            if "automation" in suggestion_data:
                automation = suggestion_data["automation"]
                alias = automation.get("alias", "Unknown automation")
                description = automation.get("description", "")
                
                # Create a more natural response
                response = f"{message_text}\n\n**{alias}**"
                if description:
                    response += f"\n{description}"
                
                # Add trigger info if available
                if "trigger" in automation and automation["trigger"]:
                    trigger = automation["trigger"][0]  # Get first trigger
                    if trigger.get("platform") == "time":
                        response += f"\n\nTrigger: Daily at {trigger.get('at', 'unknown time')}"
                    elif trigger.get("platform") == "state":
                        entity = trigger.get("entity_id", "entity")
                        response += f"\nTrigger: When {entity} changes"
                    elif trigger.get("platform") == "device":
                        response += f"\nTrigger: Device trigger"
                
                # Add action info
                if "action" in automation and automation["action"]:
                    action = automation["action"][0]  # Get first action
                    if "service" in action:
                        service = action["service"]
                        target = action.get("target", {}).get("entity_id", "entity")
                        response += f"\nAction: {service} on {target}"
                
                response += "\n\nSay 'yes' or 'create it' to create this automation, or 'no' to cancel."
                
                return response
            else:
                return message_text
                
        except Exception as e:
            _LOGGER.error("Error formatting automation suggestion: %s", e)
            return str(suggestion_data)

    def _format_single_entity_response(self, entity_data: dict) -> str:
        """Format a single entity response into readable text."""
        try:
            _LOGGER.warning("DEBUG CONV: Formatting single entity: %s", entity_data)
            
            # Check if it's entity information
            if "entity_id" in entity_data and "state" in entity_data:
                entity_id = entity_data["entity_id"]
                state = entity_data["state"]
                friendly_name = entity_data.get("friendly_name") or entity_data.get("attributes", {}).get("friendly_name", entity_id)
                
                # Format based on entity type
                domain = entity_id.split(".")[0] if "." in entity_id else "unknown"
                
                if domain == "switch":
                    state_text = "on" if state == "on" else "off"
                    return f"The {friendly_name} switch is {state_text}."
                elif domain == "light":
                    state_text = "on" if state == "on" else "off"
                    return f"The {friendly_name} light is {state_text}."
                elif domain == "sensor":
                    unit = entity_data.get("attributes", {}).get("unit_of_measurement", "")
                    return f"The {friendly_name} sensor reads {state}{' ' + unit if unit else ''}."
                elif domain == "binary_sensor":
                    state_text = "active" if state == "on" else "inactive"
                    return f"The {friendly_name} sensor is {state_text}."
                elif domain == "climate":
                    temp = entity_data.get("attributes", {}).get("current_temperature", state)
                    return f"The {friendly_name} thermostat is at {temp}°."
                else:
                    return f"The {friendly_name} is {state}."
            
            # If it's not entity data, try to format it generically
            else:
                return self._format_dict_response(entity_data)
                
        except Exception as e:
            _LOGGER.error("Error formatting single entity response: %s", e)
            return f"Entity information: {entity_data}"

    def _format_list_response(self, data_list: list) -> str:
        """Format a list response into readable text."""
        try:
            if not data_list:
                return "No items found."
            
            _LOGGER.warning("DEBUG CONV: Formatting list of %d items", len(data_list))
            _LOGGER.warning("DEBUG CONV: First item keys: %s", list(data_list[0].keys()) if data_list and isinstance(data_list[0], dict) else "not dict")
            
            # Check if it's a list of automations (has entity_id starting with automation.)
            if all(isinstance(item, dict) and "friendly_name" in item for item in data_list):
                # Check if they are automations by entity_id or if they have automation-like fields
                has_automation_entity = any(
                    item.get("entity_id", "").startswith("automation.") 
                    for item in data_list 
                    if "entity_id" in item
                )
                has_state = any("state" in item for item in data_list)
                
                if has_automation_entity or has_state:
                    # It's a list of automations
                    response = f"Here are your {len(data_list)} automations:\n\n"
                    
                    for i, automation in enumerate(data_list, 1):
                        name = automation.get("friendly_name", "Unknown")
                        state = automation.get("state", "unknown")
                        
                        # Try multiple time fields
                        last_time = (
                            automation.get("last_triggered") or 
                            automation.get("last_changed") or 
                            automation.get("last_updated") or 
                            "never"
                        )
                        
                        # Format last activity time
                        if last_time and last_time != "never":
                            try:
                                from datetime import datetime
                                if isinstance(last_time, str) and "T" in last_time:
                                    dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                                    last_time = dt.strftime("%m-%d at %H:%M")
                                else:
                                    last_time = str(last_time)[:16]  # Truncate if too long
                            except Exception as e:
                                _LOGGER.debug("Error parsing datetime %s: %s", last_time, e)
                                last_time = "unknown"
                        
                        status_emoji = "✅" if state == "on" else "❌"
                        response += f"{i}. {status_emoji} **{name}**\n"
                        response += f"   State: {state.title()}\n"
                        if last_time and last_time != "never":
                            response += f"   Last activity: {last_time}\n"
                        response += "\n"
                    
                    final_response = response.strip()
                    _LOGGER.warning("DEBUG CONV: Returning automation list, length: %d", len(final_response))
                    return final_response
                else:
                    # List of entities but not automations
                    response = f"Found {len(data_list)} entities:\n\n"
                    
                    for i, entity in enumerate(data_list, 1):
                        name = entity.get("friendly_name", entity.get("name", f"Item {i}"))
                        state = entity.get("state", "unknown")
                        response += f"{i}. **{name}**: {state}\n"
                    
                    return response.strip()
            
            # Check if it's a list of other entities
            elif all(isinstance(item, dict) and "entity_id" in item for item in data_list):
                response = f"Found {len(data_list)} entities:\n\n"
                
                for i, entity in enumerate(data_list, 1):
                    entity_id = entity.get("entity_id", "unknown")
                    state = entity.get("state", "unknown")
                    friendly_name = entity.get("friendly_name") or entity.get("attributes", {}).get("friendly_name", entity_id)
                    
                    response += f"{i}. **{friendly_name}**: {state}\n"
                
                return response.strip()
            
            # Generic list formatting - ensure we always return a string
            else:
                response = f"Found {len(data_list)} items:\n\n"
                
                for i, item in enumerate(data_list, 1):
                    if isinstance(item, dict):
                        # Try to find a meaningful display field
                        display_value = (
                            item.get("name") or 
                            item.get("friendly_name") or 
                            item.get("title") or 
                            item.get("alias") or
                            str(item)[:50]  # Limit length for safety
                        )
                        response += f"{i}. {display_value}\n"
                    else:
                        response += f"{i}. {str(item)[:50]}\n"  # Limit length for safety
                
                return response.strip()
                
        except Exception as e:
            _LOGGER.error("Error formatting list response: %s", e)
            # ALWAYS return a string, never None
            return f"Found {len(data_list)} items. Error formatting: {str(e)}"

    def _format_dict_response(self, data: dict) -> str:
        """Format a dictionary response into readable text."""
        try:
            # Handle automation suggestions specifically
            if data.get("request_type") == "automation_suggestion":
                message = data.get("message", "I've created an automation suggestion.")
                
                if "automation" in data:
                    automation = data["automation"]
                    alias = automation.get("alias", "Unknown automation")
                    description = automation.get("description", "")
                    
                    response = f"{message}\n\n**{alias}**"
                    if description:
                        response += f"\n{description}"
                    
                    return response
                else:
                    return message
            
            # If it looks like an entity state response
            elif "entity_id" in data and "state" in data:
                entity_id = data["entity_id"]
                state = data["state"]
                friendly_name = data.get("attributes", {}).get("friendly_name", entity_id)
                return f"{friendly_name} is {state}"
            
            # If it contains automation or action info
            elif "automation" in data or "action" in data:
                return "I've processed your automation request successfully."
            
            # If it contains dashboard info
            elif "dashboard" in data or "url" in data:
                return "I've created/updated the dashboard successfully."
            
            # Handle success responses
            elif "success" in data and data["success"]:
                if "answer" in data:
                    return str(data["answer"])
                elif "message" in data:
                    return str(data["message"])
                else:
                    return "Request completed successfully."
            
            # Generic formatting for other dict responses
            elif len(data) == 1:
                key, value = next(iter(data.items()))
                return f"{key}: {value}"
            
            # For complex dicts, create a readable summary
            else:
                summary_parts = []
                for key, value in data.items():
                    if key not in ["timestamp", "id", "metadata", "request_type"]:  # Skip technical fields
                        if isinstance(value, str) and len(value) < 200:  # Reasonable length
                            summary_parts.append(f"{key}: {value}")
                        elif not isinstance(value, dict):  # Skip complex nested objects
                            summary_parts.append(f"{key}: {value}")
                
                return "; ".join(summary_parts) if summary_parts else "Request processed successfully."
            
        except Exception as e:
            _LOGGER.error("Error formatting dict response: %s", e)
            return str(data)

    async def _direct_ai_call(self, message: str) -> str:
        """Direct call to AI client as fallback."""
        try:
            # Try to get the agent directly from the stored data
            if "agent" in self._integration_data:
                agent = self._integration_data["agent"]
                provider = self._integration_data.get("provider", "openai")
                direct_result = await agent.process_query(message, provider=provider)
                
                # Format the direct result the same way
                if isinstance(direct_result, dict):
                    if "request_type" in direct_result and direct_result["request_type"] == "automation_suggestion":
                        return "I've created an automation suggestion. Please check the AI Agent HA dashboard to review and create it."
                    elif "answer" in direct_result:
                        return str(direct_result["answer"])
                    elif "response" in direct_result:
                        return str(direct_result["response"])
                    else:
                        return self._format_dict_response(direct_result)
                elif isinstance(direct_result, list):
                    return self._format_list_response(direct_result)
                else:
                    return str(direct_result)
            else:
                raise HomeAssistantError("No AI agent found in integration data")
                
        except Exception as err:
            _LOGGER.error("Direct AI call failed: %s", err)
            raise HomeAssistantError(f"AI processing failed: {err}")

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.config_entry.entry_id)},
            "name": "AI Agent HA",
            "manufacturer": "AI Agent HA",
            "model": self.config_entry.data.get('ai_provider', 'Unknown Provider'),
            "sw_version": "1.0.0",
        }