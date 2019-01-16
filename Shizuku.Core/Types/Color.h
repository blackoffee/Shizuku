#pragma once

#include <glm/glm.hpp>

#ifdef SHIZUKU_CORE_EXPORTS  
#define CORE_API __declspec(dllexport)   
#else  
#define CORE_API __declspec(dllimport)   
#endif  

namespace Shizuku{ namespace Core{ namespace Types{
	class Color
	{
	private:
		glm::vec4 m_color;

	public:
		Color()
		{
		}

		Color(const glm::vec4& p_rgba)
			:m_color(p_rgba)
		{
		}

		Color(const glm::uvec4& p_rgba)
			:m_color(glm::vec4(p_rgba)/255.f)
		{
		}

		const glm::vec4& Value() const
		{
			return m_color;
		}
	};
} } }
