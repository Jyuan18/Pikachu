## 第一章 起步
### 1.1 版本控制
#### 1.1.1 本地版本控制系统
#### 1.1.2 集中化的版本控制系统
#### 1.1.3 分布式版本控制系统

### 1.2 git历史

### 1.3 git基础要点
#### 1.3.1 直接快照,而非比较差异
> git只关心文件系统的整体是否变化,不关心文件内容的前后差异
#### 1.3.2 几乎所有操作都可本地执行
#### 1.3.3 时刻保持数据完整性
> git使用sha-1算法计算数据的校验和,通过对文件内容或目录结构计算一个sha-1哈希值,作为指纹字符串
#### 1.3.4 多数操作仅仅添加数据
#### 1.3.5 三种状态
> git内部文件的三种状态:已提交committed\已修改modified\已暂存staged
- 已提交表示该文件已经被安全地保存在本地数据库中
- 已修改表示修改了某个文件,还没有提交保存
- 已暂存表示把已修改的文件放在下次提交时要保存的清单中

### 1.4 安装git
#### 1.4.1 从源代码安装
#### 1.4.2 在linux上安装
#### 1.4.3 在mac上安装
#### 1.4.4 在windows上安装

### 1.5 初始配置
> git config工具
- 系统配置:git config --system
- 用户全局配置:git config --global
- 项目配置:.git/config
#### 1.5.1 用户信息
`git config --global user.name 'xxx'`
`git config --global user.email 'xxx@xxx.com'`
如果用了--global选项,那么更改的配置文件就是位于用户主目录下,所有项目默认使用该配置
如果要在某个特定项目中使用,只要去掉--global选项重新配置,新的设定保存在当前项目的.git/config文件里
#### 1.5.2 文本编辑器
git需要输入一些额外信息时,会自动调用一个外部文本编辑器使用,默认使用系统默认编辑器如vi或vim
`git config --global core.editor vim`
#### 1.5.2 差异分析工具
`git config --global merge.tool vimdiff`
#### 1.5.3 查看配置信息
```git config -list```
### 1.6 获取帮助
`git help <verb>`
`git <verb> --help`
`git help config`


## 第二章 git基础
> 最常用的git命令
### 2.1 取得项目的git仓库
#### 2.1.1 方法一:从当前目录初始化
`git init`
`git add .`
`git commit -m 'initial project version`
#### 2.1.2 方法二:从现有仓库克隆
`git clone [url]`
`git clone [url] myname`

### 2.2 记录每次更新到仓库
获取仓库->修改文件->提交仓库
> 工作目录下所有文件只有两种状态:已跟踪或未跟踪(注意区分git文件的三种状态)
- 已跟踪:本来就被纳入版本控制管理的文件,上次文件快照中有他们的记录,工作一段时间后,它们的状态可能是未更新,已修改或已放入暂存区
- 未跟踪:既没有上次更新时的快照,也不在当前的暂存区域
> 编辑过文件后git将这些文件标记为已修改,逐步把这些修改过的文件放到暂存区域,然后等最后一次性提交暂存区域的所有文件更新
#### 2.2.1 检查文件状态
`git status`
#### 2.2.2 跟踪新文件
`git add README`
git add后接要跟踪的文件或目录的路径,如果是目录的话要说明递归跟踪
#### 2.2.3 暂存已修改文件
`git add README`
#### 2.2.4 忽略某些文件
> 有些文件无需纳入git管理,也不希望出现在未跟踪文件列表
- 创建一个.gitignore文件,列出忽略的文件模式
- .gitignore格式规范
  - 所有空行或者以注释符号#开头的行会被git忽略
  - 可以使用标准的glob模式匹配
  - 匹配模式最后跟反斜杠/说明要忽略的目录
  - 要忽略指定模式以外的文件或目录,可以在模式前加上!取反
- glob模式:shell使用的简化了的正则表达式
  - 星号*匹配零个或多个任意字符
  - [abc]匹配任何一个列在括号中字符,这个例子表示匹配一个a或者一个b或者一个c
  - 问号?只匹配一个任意字符
  - [0-9]用短线分割两个字符表明所有这两个字符范围内的都可以匹配
#### 2.2.5 查看已暂存和未暂存的更新
`git status`仅列出修改过的文件,如果要查看具体修改了什么地方,用git diff命令
`git diff`查看尚未暂存的文件更新了哪些部分,即修改之后还没暂存起来的变化内容
`git diff --staged`查看已经暂存起来的文件和上次提交时的快照之间的差异
#### 2.2.6 提交更新
`git commit`
`git commit -m 'explaination'`
#### 2.2.7 跳过使用暂存区域
`git commit -a -m 'explaination'`
#### 2.2.8 移除文件
> 移除某个文件要从已跟踪文件清单中移除(即从暂存区域移除),然后提交
`git rm xxx`
`git commit -m 'rm xxx'`
`git rm -f xxx'`如果删除之前修改过并且已经放到暂存区域
`git rm -r -f xxx`
> 把文件从git仓库中删除,但是仍然希望保留在当前工作目录中,例如移除日志文件跟踪但不删除文件,稍后在.gitignore中补上
`git rm --cached xxx`
#### 2.2.9 移动文件-改名
`git mv old new`

### 2.3 查看提交历史
`git log`默认不用任何参数,git log会按提交时间列出所有的更新,最近的更新排在最上面
`git log -p -2`-p选项展开最近每次提交的内容差异,-2仅显示最近的两次更新
`git log --stat`仅显示简要的增改行数统计
`git log --pretty=oneline`short\full\fuller 通过不同选项展示不同的提交信息
`git log --pretty=format:"%h - %an, %ar : %s"`

![alt text](https://github.com/Jyuan18/Pikachu/blob/main/assets/image.png)
#### 2.3.1 限制输出长度
`git log --since=2.weeks`
#### 2.3.2 使用图形化工具查阅历史

### 2.4 撤销操作
#### 2.4.1 修改最后一次提交
`git commit --amend`
#### 2.4.2 取消已经暂存的文件
`git reset HEAD xxx`
#### 2.4.3 取消对文件的修改
`git checkout xxx`

### 2.5 远程仓库的使用
远程仓库是指托管在网络上的项目仓库,可能会有好多个,协作需要推送或拉取数据
#### 2.5.1 查看当前的远程仓库
`git remote`列出每个远程库的简短名字,克隆完某个项目后,至少可以看到一个名为origin的远程库,git默认使用这个名字标识克隆的原始仓库
`git remote -v`--verbose的简写,显示对应的克隆地址
#### 2.5.2 添加远程仓库
`git remote add [shortname] [url]`
#### 2.5.3 从远程仓库抓取数据到本地
`git fetch [remote-name]`
#### 2.5.4 推送数据到远程仓库
`git push [remote-name] [branch-name]`把本地的branch-Name分支推送到remote-name服务器上
#### 2.5.5 查看远程仓库信息
`git remote show origin`
#### 2.5.6 远程仓库的删除和重命名
`git remote rename name1 name2`
`git remote rm xxx`删除远端仓库

### 2.6 打标签
#### 2.6.1 列显已有的标签
`git tag`
`git tag -1 'v1.4.*'`
#### 2.6.2 新建标签
> 轻量级lightweight和含附注的annotated
#### 2.6.3 含附注的标签
`git tag -a v1.2 -m 'xxx'`
`git show v1.2`
#### 2.6.4 签署标签
`git tag -s v1.2 -m 'xxx'`
#### 2.6.5 轻量级标签
`git tag v1.4-1w`
#### 2.6.6 验证标签
`git tag -v v1.2`
#### 2.6.7 后期加注标签
#### 2.6.8 分享标签

### 2.7 技巧和窍门
