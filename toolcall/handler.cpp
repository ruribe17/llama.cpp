
#include <json.hpp>
#include "toolcall-handler.h"
#include <chrono>
#include <stdexcept>

#ifdef LLAMA_USE_CURL
#  include "mcp_sse_transport.h"
#endif

#include "mcp_stdio_transport.h"

using json = nlohmann::json;

std::shared_ptr<toolcall::handler> toolcall::create_handler(const toolcall::params & params) {
    std::shared_ptr<toolcall::handler> handler;

    auto tools = params.tools();
    auto choice = params.choice();
    if (params) {
        if (params.has_uri()) {
#ifdef LLAMA_USE_CURL
            handler.reset(new toolcall::handler(
                              std::make_unique<toolcall::mcp_impl>(tools, choice)));
#endif
        } else {
            handler.reset(new toolcall::handler(
                              std::make_unique<toolcall::loopback_impl>(tools, choice)));
        }
    }
    return handler;
}

std::string toolcall::handler::tool_list() {
    return impl_->tool_list();
}

bool toolcall::handler::tool_list_dirty() const {
    return impl_->tool_list_dirty();
}

toolcall::action toolcall::handler::call(const std::string & request, std::string & response) {
    last_action_ = impl_->call(request, response);
    return last_action_;
}

const std::string & toolcall::handler::tool_choice() const {
    return impl_->tool_choice();
}

toolcall::action toolcall::handler::last_action() const {
    return last_action_;
}

void toolcall::handler::initialize() {
    impl_->initialize();
}

#ifdef LLAMA_USE_CURL
toolcall::mcp_impl::mcp_impl(std::string server_uri, std::string tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_sse_transport(server_uri)),
      tools_("[]"),
      tools_mutex_(),
      tools_populating_(),
      next_id_(1)
{
}
#else
toolcall::mcp_impl::mcp_impl(std::string /*server_uri*/, std::string tool_choice)
    : handler_impl(tool_choice),
      transport_(nullptr),
      tools_("[]"),
      tools_mutex_(),
      tools_populating_(),
      next_id_(1)
{
}
#endif

toolcall::mcp_impl::mcp_impl(std::vector<std::string> argv, std::string tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_stdio_transport(argv))
{
}

void toolcall::mcp_impl::initialize() {
    using on_response = toolcall::callback<mcp::initialize_response>;
    using on_list_changed = toolcall::callback<mcp::tools_list_changed_notification>;

    if (transport_ == nullptr) return;
    std::unique_lock<std::mutex> lock(tools_mutex_);

    transport_->start();

    bool caps_received = false;
    mcp::capabilities caps;
    on_response set_caps = [this, &caps] (const mcp::initialize_response & resp) {
        std::unique_lock<std::mutex> lock(tools_mutex_);
        caps = resp.capabilities();
        tools_populating_.notify_one();
    };

    transport_->subscribe("set_caps", set_caps);

    mcp::initialize_request req(next_id_++);
    transport_->send(req);

    tools_populating_.wait_for(lock, std::chrono::seconds(15), [&caps_received] { return caps_received; });
    transport_->unsubscribe<mcp::initialize_response>("set_caps");

    on_list_changed update_dirty = [this] (const mcp::tools_list_changed_notification &) {
        tool_list_dirty_ = true;
    };

    bool has_tools = false;
    for (const auto & cap : caps) {
        if (cap.name == "tools") {
            has_tools = true;
            if (cap.listChanged) {
                transport_->subscribe("update_dirty", update_dirty);
            }
            break;
        }
    }
    if (! has_tools) {
        throw std::runtime_error("MCP server does not support toolcalls!");
    }

    transport_->send(mcp::initialized_notification());
}

static std::string tools_list_to_oai_json(const mcp::tools_list & tools) {
    json tool_list = json::array();
    for (const auto & tool : tools) {
        json props;
        for (const auto & param : tool.params) {
            props[param.name]["type"] = param.type;
            props[param.name]["description"] = param.description;
        }
        json required = json::array();
        for (const auto & name : tool.required_params) {
            required.push_back(name);
        }
        tool_list.push_back({
                {"type", "function"},
                {"function", {
                        {"name", tool.tool_name},
                        {"description", tool.tool_description},
                        {"parameters", {
                                {"type", "object"},
                                {"properties", props}
                            }
                        },
                        {"required", required}
                    }
                }
            });
    }
    return tool_list;
}

std::string toolcall::mcp_impl::tool_list() {
    using on_response = toolcall::callback<mcp::tools_list_response>;

    if (tool_list_dirty_) {
        std::unique_lock<std::mutex> lock(tools_mutex_);

        mcp::tools_list tools;
        on_response set_tools = [this, &tools] (const mcp::tools_list_response & resp) {
            std::unique_lock<std::mutex> lock(tools_mutex_);

            tools.insert(tools.end(), resp.tools().begin(), resp.tools().end());
            auto cursor = resp.next_cursor();
            if (! cursor.empty()) {
                mcp::tools_list_request req(next_id_++, cursor);
                transport_->send(req);
                return;
            }
            tool_list_dirty_ = false;
            lock.unlock();
            tools_populating_.notify_one();
        };

        transport_->subscribe("set_tools", set_tools);

        mcp::tools_list_request req(next_id_++);
        transport_->send(req);

        tools_populating_.wait_for(lock, std::chrono::seconds(15), [this] { return ! tool_list_dirty_; });
        transport_->unsubscribe<mcp::tools_list_response>("set_tools");

        tools_ = tools_list_to_oai_json(tools);
    }
    return tools_;
}

toolcall::action toolcall::mcp_impl::call(const std::string & /*request*/, std::string & /*response*/) {
    if (transport_ == nullptr) {
        return toolcall::DEFER;
    }
    // Construct tool call and send to transport
    return toolcall::ACCEPT; // TODO
}
