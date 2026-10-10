# Relay route: send Muse's hostnames through your own TLS relay. See Kconfig.
idf_component_register(SRCS "relay_route.c" PRIV_REQUIRES lwip log WHOLE_ARCHIVE)
if(CONFIG_RELAY_ROUTE_ENABLE)
    target_link_libraries(${COMPONENT_LIB} INTERFACE "-Wl,--wrap=lwip_connect")
endif()
