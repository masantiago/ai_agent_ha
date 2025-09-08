"""The AI Agent HA integration."""

from __future__ import annotations

import logging

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .agent import AiAgentHaAgent
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Define platforms - adding conversation support
PLATFORMS: list[Platform] = [Platform.CONVERSATION]

# Config schema - this integration only supports config entries
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Define service schema to accept a custom prompt
SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("prompt"): cv.string,
    }
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the AI Agent HA component."""
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Migrate old config entries to new version."""
    _LOGGER.debug("Migrating config entry from version %s", entry.version)

    if entry.version == 1:
        # No migration needed for version 1
        return True

    # Future migrations would go here
    # if entry.version < 2:
    #     # Migrate from version 1 to 2
    #     new_data = dict(entry.data)
    #     # Add migration logic here
    #     hass.config_entries.async_update_entry(entry, data=new_data, version=2)

    _LOGGER.info("Migration to version %s successful", entry.version)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AI Agent HA from a config entry."""
    try:
        # Handle version compatibility
        if not hasattr(entry, "version") or entry.version != 1:
            _LOGGER.warning(
                "Config entry has version %s, expected 1. Attempting compatibility mode.",
                getattr(entry, "version", "unknown"),
            )

        # Convert ConfigEntry to dict and ensure all required keys exist
        config_data = dict(entry.data)

        # Ensure backward compatibility - check for required keys
        if "ai_provider" not in config_data:
            _LOGGER.error(
                "Config entry missing required 'ai_provider' key. Entry data: %s",
                config_data,
            )
            raise ConfigEntryNotReady("Config entry missing required 'ai_provider' key")

        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {"agents": {}, "configs": {}}

        provider = config_data["ai_provider"]

        # Validate provider
        if provider not in [
            "llama",
            "openai",
            "gemini",
            "openrouter",
            "anthropic",
            "local",
        ]:
            _LOGGER.error("Unknown AI provider: %s", provider)
            raise ConfigEntryNotReady(f"Unknown AI provider: {provider}")

        # Store config for this provider
        hass.data[DOMAIN]["configs"][provider] = config_data

        # Create agent for this provider
        _LOGGER.debug(
            "Creating AI agent for provider %s with config: %s",
            provider,
            {
                k: v
                for k, v in config_data.items()
                if k
                not in [
                    "llama_token",
                    "openai_token",
                    "gemini_token",
                    "openrouter_token",
                    "anthropic_token",
                ]
            },
        )
        
        # Create and store the agent
        agent = AiAgentHaAgent(hass, config_data)
        hass.data[DOMAIN]["agents"][provider] = agent
        
        # Store entry data for conversation platform
        hass.data[DOMAIN][entry.entry_id] = {
            "config": config_data,
            "agent": agent,
            "provider": provider,
        }

        _LOGGER.info("Successfully set up AI Agent HA for provider: %s", provider)

    except KeyError as err:
        _LOGGER.error("Missing required configuration key: %s", err)
        raise ConfigEntryNotReady(f"Missing required configuration key: {err}")
    except Exception as err:
        _LOGGER.exception("Unexpected error setting up AI Agent HA")
        raise ConfigEntryNotReady(f"Error setting up AI Agent HA: {err}")

    # Modify the query service handler to support return_response
    async def async_handle_query(call):
        """Handle the query service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                result = {"error": "No AI agents configured"}
                return result

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    result = {"error": "No AI agents configured"}
                    return result
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            
            # Get the agent result
            agent_result = await agent.process_query(
                call.data.get("prompt", ""), provider=provider
            )
            
            # Return result for conversation agent - ai_agent_ha returns {'success': True, 'answer': 'text'}
            return agent_result
            
        except Exception as e:
            _LOGGER.error(f"Error processing query: {e}")
            result = {"error": str(e)}
            return result

    async def async_handle_create_automation(call):
        """Handle the create_automation service call."""
        try:
            # Check if agents are available
            if DOMAIN not in hass.data or not hass.data[DOMAIN].get("agents"):
                _LOGGER.error(
                    "No AI agents available. Please configure the integration first."
                )
                return {"error": "No AI agents configured"}

            provider = call.data.get("provider")
            if provider not in hass.data[DOMAIN]["agents"]:
                # Get the first available provider
                available_providers = list(hass.data[DOMAIN]["agents"].keys())
                if not available_providers:
                    _LOGGER.error("No AI agents available")
                    return {"error": "No AI agents configured"}
                provider = available_providers[0]
                _LOGGER.debug(f"Using fallback provider: {provider}")

            agent = hass.data[DOMAIN]["agents"][provider]
            result = await agent.create_automation(call.data.get("automation", {}))
            return result
        except Exception as e:
            _LOGGER.error(f"Error creating automation: {e}")
            return {"error": str(e)}



    # Register services with return_response support for conversation agent
    hass.services.async_register(
        DOMAIN, 
        "query", 
        async_handle_query,
        schema=vol.Schema({
            vol.Required("prompt"): cv.string,
            vol.Optional("provider"): cv.string,
        }),
        supports_response=SupportsResponse.OPTIONAL  # Enable return_response
    )
    
    hass.services.async_register(
        DOMAIN, 
        "create_automation", 
        async_handle_create_automation,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    


    # Set up conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    

    # Remove services only if this is the last config entry
    remaining_entries = [
        entry for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.entry_id != entry.entry_id
    ]
    
    if not remaining_entries:
        hass.services.async_remove(DOMAIN, "query")
        hass.services.async_remove(DOMAIN, "create_automation")

    # Remove entry data
    if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
        hass.data[DOMAIN].pop(entry.entry_id)
        
    # Clean up provider data if no more entries use this provider
    if entry.data.get("ai_provider") in hass.data[DOMAIN]["agents"]:
        provider_in_use = any(
            e.data.get("ai_provider") == entry.data.get("ai_provider")
            for e in remaining_entries
        )
        if not provider_in_use:
            hass.data[DOMAIN]["agents"].pop(entry.data.get("ai_provider"))
            hass.data[DOMAIN]["configs"].pop(entry.data.get("ai_provider"))

    return unload_ok


