#pragma once

#include "toolcall-params.h"
#include <string>
#include <variant>
#include <memory>
#include <vector>
#include <condition_variable>
#include <mutex>

namespace toolcall
{
    enum action {
        ACCEPT,
        PENDING,
        DEFER
    };

    class handler_impl;
    class handler {
    public:
        using ptr = std::shared_ptr<handler>;

        handler(std::unique_ptr<handler_impl> impl) : impl_(std::move(impl)) {}

        action call(const std::string & request, std::string & response);

        std::string tool_list();
        bool tool_list_dirty() const;

        const std::string & tool_choice() const;
        action last_action() const;

        void initialize();

    private:
        std::unique_ptr<handler_impl> impl_;
        action last_action_;
    };

    std::shared_ptr<toolcall::handler> create_handler(const toolcall::params & params);

    class handler_impl {
    public:
        handler_impl(std::string tool_choice)
            : tool_choice_(std::move(tool_choice)), tool_list_dirty_(true) {}

        virtual ~handler_impl() = default;

        virtual std::string tool_list() = 0;

        virtual bool tool_list_dirty() const {
            return tool_list_dirty_;
        }

        virtual action call(const std::string & request, std::string & response) = 0;

        const std::string & tool_choice() const { return tool_choice_; }

        virtual void initialize() {}

    protected:
        std::string tool_choice_;
        bool tool_list_dirty_;
    };

    class loopback_impl : public handler_impl {
    public:
        loopback_impl(std::string tools, std::string tool_choice)
            : handler_impl(tool_choice), tools_(std::move(tools)) {}

        virtual std::string tool_list() override {
            tool_list_dirty_ = false;
            return tools_;
        }

        virtual action call(const std::string & request, std::string & response) override {
            response = request;
            return toolcall::DEFER;
        }

    private:
        std::string tools_;
    };

    class mcp_transport;
    class mcp_impl : public handler_impl {
    public:
        mcp_impl(std::string server_uri, std::string tool_choice);
        mcp_impl(std::vector<std::string> argv, std::string tool_choice);

        virtual std::string tool_list() override;
        virtual action call(const std::string & request, std::string & response) override;

        virtual void initialize() override;

    private:
        std::unique_ptr<mcp_transport> transport_;
        std::string tools_;
        std::mutex tools_mutex_;
        std::condition_variable tools_populating_;
        int next_id_;
    };
}
