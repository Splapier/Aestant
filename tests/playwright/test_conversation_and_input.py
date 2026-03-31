"""Playwright tests verifying the conversation box and prompt input textbox.

Tests that the conversation box is visible and displays the conversation label,
and that the prompt input textbox is visible, enabled, and accepts user typing.
"""

from helpers import navigate


class TestConversationBox:
    """Verify the conversation chatbot component is visible to the user."""

    def test_conversation_box_is_visible(self, page, app_url):
        """The conversation box must be visible when the app loads.

        Steps:
        1. Navigate to the app
        2. Locate the chatbot component labelled 'Conversation'
        3. Assert it is visible on the page
        """
        navigate(page, app_url)

        chatbot = page.locator("label:has-text('Conversation')").first
        chatbot.wait_for(state="visible", timeout=5000)
        assert chatbot.is_visible(), "Conversation box should be visible"

    def test_conversation_box_shows_label(self, page, app_url):
        """The conversation box must display the 'Conversation' label text.

        Steps:
        1. Navigate to the app
        2. Locate the chatbot wrapper element
        3. Assert it contains the 'Conversation' text
        """
        navigate(page, app_url)

        chatbot = page.locator("label:has-text('Conversation')").first
        chatbot.wait_for(state="visible", timeout=5000)
        assert "Conversation" in chatbot.text_content(), (
            "Conversation box should display the 'Conversation' label"
        )


class TestPromptInput:
    """Verify the prompt input textbox is visible and accepts user typing."""

    def test_prompt_input_is_visible_and_enabled(self, page, app_url):
        """The prompt input textbox must be visible and enabled for the user.

        Steps:
        1. Navigate to the app
        2. Locate the textbox labelled 'Your Message'
        3. Assert it is visible and not disabled
        """
        navigate(page, app_url)

        textbox = page.get_by_label("Your Message")
        textbox.wait_for(state="visible", timeout=5000)
        assert textbox.is_visible(), "Prompt input textbox should be visible"
        assert not textbox.is_disabled(), "Prompt input textbox should be enabled"

    def test_prompt_input_accepts_user_typing(self, page, app_url):
        """The prompt input textbox must accept and display typed text.

        Steps:
        1. Navigate to the app
        2. Locate the textbox labelled 'Your Message'
        3. Type a message into it
        4. Assert the textbox value matches the typed text
        """
        navigate(page, app_url)

        textbox = page.get_by_label("Your Message")
        textbox.wait_for(state="visible", timeout=5000)

        test_message = "Hello, this is a test message"
        textbox.fill(test_message)

        assert textbox.input_value() == test_message, (
            "Prompt input textbox should contain the typed message"
        )
