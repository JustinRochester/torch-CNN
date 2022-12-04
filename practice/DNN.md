
$$
\\\boldsymbol h_1&=&f(\bold W_{i, 1}\cdot \boldsymbol i+\boldsymbol \beta_1)
\\\boldsymbol h_2&=&f(\bold W_{1, 2}\cdot \boldsymbol h_1+\boldsymbol \beta_2)
\\\boldsymbol o&=&\bold W_{2, o}\cdot \boldsymbol h_2+\boldsymbol \beta_o
\\L&=&\sqrt{{1\over N}\cdot \sum_i(\boldsymbol o_i-\boldsymbol y_i)^T\cdot (\boldsymbol o_i-\boldsymbol y_i)}
\\R&=&R(\bold W_{i, 1})+R(\boldsymbol \beta_1)+R(\bold W_{1, 2})+R(\boldsymbol \beta_2)+R(\bold W_{2, o})+R(\boldsymbol \beta_o)
\\R(\bold A)&=&{1\over 2}\sum_i\sum_j (\bold A^{(i, j)})^2
\\Loss&=&L+\alpha\cdot R
$$

$$
\\{\partial L\over \partial \boldsymbol o^{(x)}}&=&{\partial L\over \partial L^2}\cdot {\partial L^2\over \partial \boldsymbol o^{(x)}}
\\ &=& {1\over 2L}\cdot 2(\boldsymbol o^{(x)}-\boldsymbol y^{(x)})
\\{\partial L\over \partial \boldsymbol o}&=&{1\over L}\cdot (\boldsymbol o-\boldsymbol y)
$$



$$
\\{\partial Loss\over \partial \bold W_{2, o}^{(x, y)}}&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}{\partial \boldsymbol o^{(z)}\over \partial \bold W_{2, o}^{(x, y)}}+{\partial R\over \partial \bold W_{2, o}^{(x, y)}}\cdot \alpha
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}} \cdot {\partial \over \partial \bold W_{2, o}^{(x, y)}}(\sum_t \bold W_{2, o}^{(z, t)}\boldsymbol h_2^{(t)}+\boldsymbol \beta_o^{(z)})+\bold W_{2, o}^{(x, y)}\cdot\alpha
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}\cdot [z=x]\cdot \boldsymbol h_2^{(y)}+\bold W_{2, o}^{(x, y)}\cdot\alpha
\\&=&{\partial L\over \partial \boldsymbol o^{(x)}}\cdot \boldsymbol h_2^{(y)}+\bold W_{2, o}^{(x, y)}\cdot\alpha
\\{\partial Loss\over \partial \bold W_{2, o}}&=&\sum_x\sum_y \boldsymbol e_x\cdot {\partial L\over \partial \boldsymbol o^{(x)}}\cdot \boldsymbol h_2^{(y)}\cdot \boldsymbol e_y^T+\bold W_{2, o}\cdot\alpha
\\&=&{\partial L\over \partial \boldsymbol o}\cdot \boldsymbol h_2^T+\bold W_{2, o}\cdot\alpha
$$


$$
\\{\partial Loss\over \partial \boldsymbol h_2^{(x)}}&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}{\partial \boldsymbol o^{(z)}\over \partial \boldsymbol h_2^{(x)}}
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}} \cdot {\partial \over \partial \boldsymbol h_2^{(x)}}(\sum_t \bold W_{2, o}^{(z, t)}\boldsymbol h_2^{(t)}+\boldsymbol \beta_o^{(z)})
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}\cdot \bold W_{2, o}^{(z, x)}
\\{\partial Loss\over \partial \boldsymbol h_2}&=&\sum_x\boldsymbol e_x\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}\cdot \bold W_{2, o}^{(z, x)}
\\&=&(\sum_x\sum_z \boldsymbol e_z^T{\partial L\over \partial \boldsymbol o^{(z)}}\cdot \boldsymbol e_z\bold W_{2, o}^{(z, x)}\boldsymbol e_x^T)^T
\\&=&[({\partial L\over \partial \boldsymbol o})^T\cdot \bold W_{2, o}]^T
\\&=&\bold W_{2, o}^T\cdot {\partial L\over \partial \boldsymbol o}
$$

$$
\\{\partial Loss\over \partial \boldsymbol \beta_o^{(x)}}&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}{\partial \boldsymbol o^{(z)}\over \partial \boldsymbol \beta_o^{(x)}}+{\partial L\over \partial \boldsymbol \beta_o^{(x)}}\cdot\alpha
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}} \cdot {\partial \over \partial \boldsymbol \beta_o^{(x)}}(\sum_t \bold W_{2, o}^{(z, t)}\boldsymbol h_2^{(t)}+\boldsymbol \beta_o^{(z)})+\boldsymbol \beta_o^{(x)}\cdot\alpha
\\&=&\sum_z {\partial L\over \partial \boldsymbol o^{(z)}}\cdot [z=x]+\boldsymbol \beta_o^{(x)}\cdot\alpha
\\&=&{\partial L\over \partial \boldsymbol o^{(x)}}+\boldsymbol \beta_o^{(x)}\cdot\alpha
\\{\partial Loss\over \partial \boldsymbol \beta_o}&=&\sum_x \boldsymbol e_x\cdot {\partial L\over \partial \boldsymbol o^{(x)}}+\boldsymbol \beta_o\cdot\alpha
\\&=&{\partial L\over \partial \boldsymbol o}+\boldsymbol \beta_o\cdot\alpha
$$


$$
\\{\partial L\over \partial \boldsymbol i_2^{(x)}}&=&\sum_{z}{\partial L\over \partial \boldsymbol h_2^{(z)}}\cdot {\partial \over \partial \boldsymbol i_2^{(x)}}f(\boldsymbol i_2^{(z)})
\\&=&\sum_z {\partial L\over \partial \boldsymbol h_2^{(z)}} [z=x]\cdot f'(\boldsymbol i_2^{(z)})
\\&=&{\partial L\over \partial \boldsymbol h_2^{(x)}}f'(\boldsymbol i_2^{(x)})
\\{\partial L\over \partial \boldsymbol i_2}&=&{\partial L\over \partial \boldsymbol h_2}\odot f'(\boldsymbol i_2)
$$


$$
\\{\partial Loss\over \partial \bold W_{1, 2}^{(x, y)}}&=&\sum_z {\partial L\over \partial \boldsymbol i_2^{(z)}}{\partial \boldsymbol i_2^{(z)}\over \partial \bold W_{1, 2}^{(x, y)}}+\bold W_{1, 2}^{(x, y)}\cdot \alpha
\\&=&{\partial L\over \partial \boldsymbol i_2^{(x)}}\cdot \boldsymbol h_1^{(y)}+\bold W_{1, 2}^{(x, y)}\cdot \alpha
\\{\partial Loss\over \partial \bold W_{1, 2}}&=&\sum_x\sum_y \boldsymbol e_x\cdot {\partial L\over \partial \boldsymbol i_2^{(x)}}\cdot \boldsymbol h_1^{(y)}\cdot \boldsymbol e_y^T+\bold W_{1, 2}\cdot \alpha
\\&=&{\partial L\over \partial \boldsymbol i_2}\cdot \boldsymbol h_1^T+\bold W_{1, 2}\cdot \alpha
$$
