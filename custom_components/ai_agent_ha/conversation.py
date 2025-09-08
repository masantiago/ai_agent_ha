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

# Multilingual confirmation messages
CONFIRMATION_MESSAGES = {
    "es": "Di 'sí' o 'créala' para crear esta automatización.",
    "en": "Say 'yes' or 'create it' to create this automation.",
    "fr": "Dites 'oui' ou 'créez-la' pour créer cette automatisation.",
    "de": "Sagen Sie 'ja' oder 'erstellen' um diese Automatisierung zu erstellen.",
    "it": "Di 'sì' o 'creala' per creare questa automazione.",
    "pt": "Diga 'sim' ou 'criar' para criar esta automação."
}

# Multilingual success messages
SUCCESS_MESSAGES = {
    "es": "¡Automatización creada exitosamente!",
    "en": "Automation created successfully!",
    "fr": "Automatisation créée avec succès !",
    "de": "Automatisierung erfolgreich erstellt!",
    "it": "Automazione creata con successo!",
    "pt": "Automação criada com sucesso!"
}

# Multilingual error messages
ERROR_MESSAGES = {
    "es": "Error al crear la automatización: {error}",
    "en": "Error creating automation: {error}",
    "fr": "Erreur lors de la création de l'automatisation : {error}",
    "de": "Fehler beim Erstellen der Automatisierung: {error}",
    "it": "Errore nella creazione dell'automazione: {error}",
    "pt": "Erro ao criar automação: {error}"
}

# Multilingual cancellation messages
CANCELLATION_MESSAGES = {
    "es": "Creación de automatización cancelada.",
    "en": "Automation creation cancelled.",
    "fr": "Création d'automatisation annulée.",
    "de": "Automatisierungserstellung abgebrochen.",
    "it": "Creazione automazione annullata.",
    "pt": "Criação de automação cancelada."
}

def detect_language_from_message(message: str) -> str:
    """Detect language from message content."""
    message_lower = message.lower()
    
    # Spanish indicators
    spanish_indicators = ['automatización', 'interruptor', 'apagar', 'encender', 'todos los días', 
                         'de la noche', 'de la mañana', 'tarde', 'noche', 'mañana', 'enciende', 
                         'apaga', 'celia', 'salón', 'cocina', 'dormitorio', 'baño', 'luz', 'luces']
    
    # French indicators  
    french_indicators = ['automatisation', 'interrupteur', 'éteindre', 'allumer', 'tous les jours',
                        'du soir', 'du matin', 'lumière', 'lumières']
    
    # German indicators
    german_indicators = ['automatisierung', 'schalter', 'ausschalten', 'einschalten', 'jeden tag',
                        'abends', 'morgens', 'licht', 'lichter']
    
    # Italian indicators
    italian_indicators = ['automazione', 'interruttore', 'spegnere', 'accendere', 'ogni giorno',
                         'di sera', 'di mattina', 'luce', 'luci']
    
    # Portuguese indicators
    portuguese_indicators = ['automação', 'interruptor', 'desligar', 'ligar', 'todos os dias',
                           'da noite', 'da manhã', 'luz', 'luzes']
    
    # Count matches for each language
    spanish_matches = sum(1 for indicator in spanish_indicators if indicator in message_lower)
    french_matches = sum(1 for indicator in french_indicators if indicator in message_lower)
    german_matches = sum(1 for indicator in german_indicators if indicator in message_lower)
    italian_matches = sum(1 for indicator in italian_indicators if indicator in message_lower)
    portuguese_matches = sum(1 for indicator in portuguese_indicators if indicator in message_lower)
    
    # Return language with most matches, default to English
    max_matches = max(spanish_matches, french_matches, german_matches, italian_matches, portuguese_matches)
    
    if max_matches == 0:
        return "en"  # Default to English
    
    if spanish_matches == max_matches:
        return "es"
    elif french_matches == max_matches:
        return "fr"
    elif german_matches == max_matches:
        return "de"
    elif italian_matches == max_matches:
        return "it"
    elif portuguese_matches == max_matches:
        return "pt"
    
    return "en"  # Fallback to English

# Official response types defined in agent.py SYSTEM_PROMPT
OFFICIAL_REQUEST_TYPES = {
    "FINAL_RESPONSE": "final_response",
    "AUTOMATION_SUGGESTION": "automation_suggestion", 
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


class AutomationSuggestionHandler(BaseResponseHandler):
    """Handler for automation_suggestion type - official format from agent."""
    
    def __init__(self, hass: HomeAssistant, integration_data: Dict[str, Any]):
        super().__init__(hass, integration_data)
        self._pending_automation = None
        self._original_message = None  # Store original message to detect language
    
    def get_request_type(self) -> str:
        return OFFICIAL_REQUEST_TYPES["AUTOMATION_SUGGESTION"]
    
    async def handle(self, result: Dict[str, Any]) -> str:
        """Handle automation suggestion according to official format."""
        # Official format: {"request_type": "automation_suggestion", "message": "...", "automation": {...}}
        message = result.get("message", "I've created an automation suggestion.")
        automation = result.get("automation")
        
        if not automation:
            return message
        
        # Store for confirmation
        self._pending_automation = automation
        
        # Format according to official structure
        alias = automation.get("alias", "Unknown Automation")
        description = automation.get("description", "")
        
        response = f"{message}\n\n**{alias}**"
        if description:
            response += f"\n{description}"
        
        # Detect language and add appropriate confirmation message
        if self._original_message:
            detected_language = detect_language_from_message(self._original_message)
            confirmation_msg = CONFIRMATION_MESSAGES.get(detected_language, CONFIRMATION_MESSAGES["en"])
        else:
            confirmation_msg = CONFIRMATION_MESSAGES["en"]  # Default to English
        
        response += f"\n\n{confirmation_msg}"
        return response
    
    async def handle_confirmation(self) -> str:
        """Create the pending automation."""
        if not self._pending_automation:
            return "No automation to create."
        
        # Detect language for response messages
        if self._original_message:
            detected_language = detect_language_from_message(self._original_message)
        else:
            detected_language = "en"  # Default to English
        
        try:
            result = await self.hass.services.async_call(
                DOMAIN, "create_automation",
                {"automation": self._pending_automation},
                blocking=True, return_response=True
            )
            
            self._pending_automation = None  # Clear after attempt
            
            if result and result.get("success"):
                success_msg = SUCCESS_MESSAGES.get(detected_language, SUCCESS_MESSAGES["en"])
                return success_msg
            else:
                error = result.get("error", "Unknown error") if result else "No response"
                error_msg = ERROR_MESSAGES.get(detected_language, ERROR_MESSAGES["en"])
                return error_msg.format(error=error)
                
        except Exception as e:
            _LOGGER.error("Error creating automation: %s", e)
            error_msg = ERROR_MESSAGES.get(detected_language, ERROR_MESSAGES["en"])
            return error_msg.format(error=str(e))
    
    def has_pending_confirmation(self) -> bool:
        return self._pending_automation is not None
    
    def set_original_message(self, message: str) -> None:
        """Set the original message to detect language."""
        self._original_message = message




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
        
        # Initialize handlers for confirmations
        self.automation_handler = AutomationSuggestionHandler(hass, integration_data)
        
        # Agent response handlers in order
        self.handlers = [
            FinalResponseHandler(hass, integration_data),
            self.automation_handler,
            DataRequestHandler(hass, integration_data),
            CallServiceHandler(hass, integration_data)
        ]
    
    def set_original_message(self, message: str) -> None:
        """Set the original message for language detection."""
        self.automation_handler.set_original_message(message)
    
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
    
    def has_pending_automation(self) -> bool:
        return self.automation_handler.has_pending_confirmation()
    
    async def handle_automation_confirmation(self) -> str:
        return await self.automation_handler.handle_confirmation()
    
    def has_any_pending_confirmation(self) -> bool:
        return self.has_pending_automation()
    
    async def handle_confirmation(self) -> str:
        """Handle any type of pending confirmation."""
        if self.has_pending_automation():
            return await self.handle_automation_confirmation()
        else:
            return "No pending confirmations."


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
        # Check for pending confirmations first
        user_lower = user_text.lower().strip()
        
        _LOGGER.debug("Processing user input: '%s'", user_text)
        _LOGGER.debug("Has pending automation: %s", self._response_processor.automation_handler.has_pending_confirmation())
        
        # Check for automation confirmation
        if self._response_processor.automation_handler.has_pending_confirmation():
            _LOGGER.debug("Found pending automation, checking for confirmation keywords")
            if user_lower in ['yes', 'create it', 'sí', 'si', 'crear', 'creala', 'créala']:
                _LOGGER.debug("Confirmed - creating automation")
                return await self._response_processor.automation_handler.handle_confirmation()
            elif user_lower in ['no', 'cancel', 'cancelar', 'no gracias']:
                _LOGGER.debug("Cancelled - clearing pending automation")
                # Detect language for cancellation message
                original_msg = self._response_processor.automation_handler._original_message
                if original_msg:
                    detected_language = detect_language_from_message(original_msg)
                else:
                    detected_language = "en"
                
                # Clear pending automation
                self._response_processor.automation_handler._pending_automation = None
                
                # Return localized cancellation message
                cancellation_msg = CANCELLATION_MESSAGES.get(detected_language, CANCELLATION_MESSAGES["en"])
                return cancellation_msg
        
        # No pending confirmations, process with AI agent
        return await self._process_with_ai_agent(user_text)

    async def _process_with_ai_agent(self, message: str) -> str:
        """Process message using the existing AI Agent HA service."""
        try:
            # Set the original message for language detection
            self._response_processor.set_original_message(message)
            
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
            # Set the original message for language detection
            self._response_processor.set_original_message(message)
            
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