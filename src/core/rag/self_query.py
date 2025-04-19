import opik
from langchain_openai import ChatOpenAI
from opik.integrations.langchain import OpikTracer

import core.logger_utils as logger_utils
from core.config import settings
from core.rag.prompt_templates import SelfQueryTemplate

logger = logger_utils.get_logger(__name__)


class SelfQuery:
    opik_tracer = OpikTracer(tags=["SelfQuery"])

    @staticmethod
    @opik.track(name="SelQuery.generate_response")
    def generate_response(query: str) -> str | None:
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        chain = prompt | model
        chain = chain.with_config({"callbacks": [SelfQuery.opik_tracer]})

        response = chain.invoke({"question": query})
        chapter_name = response.content

        if chapter_name == "none":
            return None

        logger.info(
            "Successfully extracted the chapter name from the query.",
            chapter_name=chapter_name,
        )

        return chapter_name
