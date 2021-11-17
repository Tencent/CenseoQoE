### Contributing

欢迎参与 CenseoQoE 项目，你可以给我们提供建议，报告 bug，或者贡献代码。在参与贡献之前，请阅读以下指引。

#### 关于代码规范
CenseoQoE主要是C++和Python代码, 相关的语言开发应遵守以下代码规范。

- C++ 代码规范遵循[Google C++ 代码规范](https://google.github.io/styleguide/cppguide.html)
- Python 代码规范遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/)

#### 关于 issue

如果你对 CenseoQoE 有意见、建议或者发现了bug，欢迎通过 issue 给我们提出。提 issue 之前，请阅读以下指引。

- 搜索以往的 issue ，看是否已经提过，避免重复提出；
- 请确认你遇到的问题，是否在最新版本已被修复；
- 提出意见或建议时，请详细描述出现问题的平台、系统、版本以及具体的错误信息；
- 如果你的问题已经得到解决，请关闭你的 issue。

#### 如何加入
- 克隆仓库到本地
- 在本地创建个人开发分支：
   - 若在工蜂仓库上已有个人开发分支，则直接在本地建立追踪分支并切换到其中：`git branch --track <your_branch_name> origin/<your_branch_name>`；
   - 若在工蜂仓库上没有个人开发分支，可基于 master 分支创建、切换至新的个人开发分支，并推送到工蜂仓库中：`git branch -b <your_branch_name> && git push origin <your_branch_name>:<your_branch_name>`.
   - 注意本项目主要分为2个模块:[CenseoQoE-Algorithm](./CenseoQoE-Algorithm) 和 [CenseoQoE-SDK](./CenseoQoE-SDK)，因此分支的名称前缀应该将这2个模块区分。
- 在本地的个人开发分支中进行代码修改。
- 开发过程中：
   - 若希望看看别的同学开发了什么新功能，那么可以拉取最新的 master 代码并合并到本地个人分支中，执行：`git checkout <your_branch_name> && git pull origin master`.
   - 若希望提交本地修改到远程个人开发分支中（省略 add 和 commit 过程）: `git push`.
- 当个人负责的功能阶段性完工时，就可以将你的个人分支合并到 master 分支中了：在工蜂页面上，从你的个人分支发起合并到 master 分支的请求，然后等待代码 review 即可。

#### 贡献者名单

#### 参与贡献
欢迎提PR和issue。
