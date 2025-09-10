"""Conversation support for AI Agent HA."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


# Official response types defined in agent.py SYSTEM_PROMPT
OFFICIAL_REQUEST_TYPES = {
    "FINAL_RESPONSE": "final_response",
    "DATA_REQUEST": "data_request",
    "CALL_SERVICE": "call_service"
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation agent from a config entry."""
    agent = AIAgentConversation(hass, config_entry)
    async_add_entities([agent])


# Official Response Handler Classes based on agent.py specification
class BaseResponseHandler(ABC):
    """Base class for handling official agent response types."""
    
    def __init__(self, hass: HomeAssistant, integration_data: Dict[str, Any]):
        self.hass = hass
        self.integration_data = integration_data
    
    @abstractmethod
    def get_request_type(self) -> str:
        """Return the official request_type this handler processes."""
        pass
    
    def can_handle(self, result: Dict[str, Any]) -> bool:
        """Check if this handler can process the result based on request_type."""
        return result.get("request_type") == self.get_request_type()
    
    @abstractmethod
    async def handle(self, result: Dict[str, Any]) -> str:
        """Handle the result and return formatted response."""
        pass


class FinalResponseHandler(BaseResponseHandler):
    """Handler for final_response type - the definitive answer from AI."""
    
    def get_request_type(self) -> str:
        return OFFICIAL_REQUEST_TYPES["FINAL_RESPONSE"]
    
    async def handle(self, result: Dict[str, Any]) -> str:
        """Extract response content directly - this is the final formatted answer."""
        return result.get("response", "No response content")






class DataRequestHandler(BaseResponseHandler):
    """Handler for data_request type - requests for HA data."""
    
    def get_request_type(self) -> str:
        return OFFICIAL_REQUEST_TYPES["DATA_REQUEST"]
    
    async def handle(self, result: Dict[str, Any]) -> str:
        """This should not happen - data_request is processed by agent.py directly."""
        # If we see this, it means the agent didn't process the data request properly
        _LOGGER.warning("Received unprocessed data_request: %s", result)
        return "Data request was not processed properly by the agent."


class CallServiceHandler(BaseResponseHandler):
    """Handler for call_service type - service execution requests."""
    
    def get_request_type(self) -> str:
        return OFFICIAL_REQUEST_TYPES["CALL_SERVICE"]
    
    async def handle(self, result: Dict[str, Any]) -> str:
        """This should not happen - call_service is processed by agent.py directly."""
        # If we see this, it means the agent didn't process the service call properly
        _LOGGER.warning("Received unprocessed call_service: %s", result)
        return "Service call was not processed properly by the agent."


class AgentResponseProcessor:
    """Response processor based on agent.py specifications."""
    
    def __init__(self, hass: HomeAssistant, integration_data: Dict[str, Any]):
        self.hass = hass
        self.integration_data = integration_data
        
        # Agent response handlers in order
        self.handlers = [
            FinalResponseHandler(hass, integration_data),
            DataRequestHandler(hass, integration_data),
            CallServiceHandler(hass, integration_data)
        ]
    
    async def process_response(self, result: Any) -> str:
        """Process response using official agent format specifications."""
        if not isinstance(result, dict):
            return str(result) if result else "No response received"
        
        # Check for error first
        if "error" in result:
            return f"Error: {result['error']}"
        
        # Check if we have a valid request_type
        request_type = result.get("request_type")
        if not request_type:
            _LOGGER.warning("Received response without request_type: %s", result)
            return self._handle_malformed_response(result)
        
        # Find appropriate handler for this request_type
        for handler in self.handlers:
            if handler.can_handle(result):
                try:
                    return await handler.handle(result)
                except Exception as e:
                    _LOGGER.error("Handler %s failed: %s", handler.__class__.__name__, e)
                    return f"Error processing {request_type}: {e}"
        
        # Unknown request_type
        _LOGGER.warning("Unknown request_type: %s", request_type)
        return f"Unknown response type: {request_type}"
    
    def _handle_malformed_response(self, result: Dict[str, Any]) -> str:
        """Handle responses that don't follow official format."""
        # This should ideally not happen with a well-configured agent
        _LOGGER.warning("Malformed response (missing request_type): %s", result)
        
        # Try to extract useful information
        if "answer" in result:
            return str(result["answer"])
        elif "message" in result:
            return str(result["message"])
        elif "response" in result:
            return str(result["response"])
        else:
            return "Received malformed response from agent"


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
        
        # Initialize agent response processor
        self._response_processor = AgentResponseProcessor(hass, self._integration_data)

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages - major languages supported by AI providers."""
        # Return common languages that work well with major AI providers
        return ["en", "es", "fr", "de", "it", "pt"]

    @property
    def supported_features(self) -> ConversationEntityFeature:
        """Return supported features."""
        return ConversationEntityFeature.CONTROL

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
            response = await self._process_user_input(user_input.text)
            
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
    
    async def _process_user_input(self, user_text: str) -> str:
        """Process user input and return appropriate response."""
        _LOGGER.debug("Processing user input: '%s'", user_text)
        
        # Process directly with AI agent
        return await self._process_with_ai_agent(user_text)

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
            
            _LOGGER.debug("AI Agent service result: %s", result)
            
            # Use response processor to handle the result
            return await self._response_processor.process_response(result)
                
        except Exception as service_error:
            _LOGGER.error("Error calling ai_agent_ha.query service: %s", service_error)
            
            # Fallback: try to access the AI client directly if service fails
            try:
                return await self._direct_ai_call(message)
            except Exception as direct_error:
                _LOGGER.error("Error with direct AI call: %s", direct_error)
                raise HomeAssistantError(f"Unable to process request: {service_error}")





    async def _direct_ai_call(self, message: str) -> str:
        """Direct call to AI client as fallback."""
        try:
            # Try to get the agent directly from the stored data
            if "agent" in self._integration_data:
                agent = self._integration_data["agent"]
                provider = self._integration_data.get("provider", "openai")
                direct_result = await agent.process_query(message, provider=provider)
                
                # Use response processor to handle direct result as well
                return await self._response_processor.process_response(direct_result)
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