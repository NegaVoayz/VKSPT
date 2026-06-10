#include "scene/SceneConfig.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

void buildTransformMatrix(const glm::vec3& scale,
                          const glm::vec3& rotation,
                          const glm::vec3& translation,
                          float            out[3][4])
{
    glm::mat4 m(1.0f);
    m = glm::translate(m, translation);
    m = glm::rotate(m, glm::radians(rotation.z), glm::vec3(0, 0, 1));
    m = glm::rotate(m, glm::radians(rotation.y), glm::vec3(0, 1, 0));
    m = glm::rotate(m, glm::radians(rotation.x), glm::vec3(1, 0, 0));
    m = glm::scale(m, scale);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            out[r][c] = m[c][r];   // column-major → row-major
}
