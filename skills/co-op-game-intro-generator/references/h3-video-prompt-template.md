# H3 Video Prompt Template

## Prompt principle
Use the same method as the GPT confirmation-image prompt:

**Fixed video event framework + dynamic user-style fill + locked character identity + palette-linked UI system.**

The video prompt must not blindly preserve the source prompt's default style. It must preserve the timeline, UI events, player positions, equipment logic, and text structure, while dynamically rewriting visual treatment, palette, character rendering, lighting, UI surface, icons, and city-world style according to the user's selected style.

## Priority order
1. User-selected style: {visual_style}
2. Confirmed image / UI reference: {ui_ref}
3. PLAYER 1 identity reference: {player1_ref}
4. PLAYER 2 identity reference: {player2_ref}
5. Height/body comparison reference: {height_ref}
6. Fixed source video event framework

## Reference roles
- {ui_ref}: confirmed first image. Use it to lock UI layout, menu hierarchy, color system, typography scale, button structure, character-game integration, and overall composition logic.
- {player1_ref}: PLAYER 1 identity anchor. Lock exact face, hairstyle, glasses if present, facial proportions, body identity, and nickname mapping to {player1_name}.
- {player2_ref}: PLAYER 2 identity anchor. Lock exact face, hairstyle, facial proportions, body identity, and nickname mapping to {player2_name}.
- {height_ref}: body comparison anchor. Lock the visible contrast between the two players and prevent identical body proportions.

## Global style baseline
整体画风必须优先服从用户选择：{visual_style}。
保留固定基准：游戏主菜单 UI、高品质游戏宣传片质感、UI 与角色深度融合、现代商业游戏 UI 设计、极强视觉冲击力、画面简洁干净、避免过度装饰。
其余视觉层面根据 {visual_style} 动态拓展：角色画风、表情气质、服装语言、色彩系统、灯光冷暖、UI材质、按钮图标、字体质感、城市加载后的世界风格。

## Palette system
根据 {visual_style} 推导视频全程色彩，但必须保持 UI 色彩联动：
- xx 色作为主体背景 / 世界主色
- xx 色作为 UI 主体颜色
- xx 色作为文字颜色
- xx 色作为功能强调色
- 红色作为危险/退出/警示提示色
- 整体颜色控制在 5 种以内
- 高对比撞色、鲜明现代、符合 {visual_style} 的色彩语言

视频中的菜单、装备面板、玩家卡、按钮、HUD、加载条、图标和文字都必须沿用同一色彩系统，不得随机新增无关颜色。

## Character identity and style lock
PLAYER 1：五官信息参考 {player1_ref}，必须保留脸、人脸比例、发型、眼镜（如有）、个人身份和 {player1_name} 昵称对应关系。表情、画风、服装和机械装备的视觉处理根据 {visual_style} 动态优化。PLAYER 1 始终位于左侧，偏高挑/修长/敏捷，装备色为功能强调色，机械爪轻量、纤细、灵活。

PLAYER 2：五官信息参考 {player2_ref}，必须保留脸、人脸比例、发型、个人身份和 {player2_name} 昵称对应关系。表情、画风、服装和机械装备的视觉处理根据 {visual_style} 动态优化。PLAYER 2 始终位于右侧，偏矮壮/宽厚/力量型，装备色为琥珀红或与危险/力量提示一致的暖色，机械拳厚重、宽大、有重量。

禁止角色身份交换、脸部互相融合、昵称交换、两人体型趋同。

## Fixed timeline framework

### [0秒–2秒] — 双人主菜单
景别/机位：高角度俯拍大全景，延续 {ui_ref} 的构图逻辑。镜头从上方轻微下压并缓慢推进。

画面内容：PLAYER 1（{player1_name}）和 PLAYER 2（{player2_name}）并排坐在画面中央，PLAYER 1 在左，PLAYER 2 在右，抬头看向摄像机。两人只有自然呼吸、眨眼和轻微身体动作。

UI结构：左上角玩家资料卡准确显示：
“PLAYER 1”
“{player1_name}”
“READY”
顶部中央或左上双卡系统中的第二张资料卡准确显示：
“PLAYER 2”
“{player2_name}”
“READY”
右侧纵向菜单准确显示：
“START NEW GAME”
“CONTINUE”
“SETTINGS”
“EXIT GAME”
“CONTINUE”是视觉中心和主高亮按钮。

动态风格填充：主菜单背景、地面纹理、按钮形态、图标、字体、边框、辉光和贴纸质感全部根据 {visual_style} 优化，但保留 {ui_ref} 的布局和层级。

声音：菜单环境音、轻微 UI hover 声、点击前的低频电子氛围。若生成无音频，则忽略声音执行。

### [2秒–4秒] — PLAYER 1 右臂配置
景别/机位：中景，镜头从主菜单平滑推近 PLAYER 1 的右臂，PLAYER 2 仍在背景中可见，不消失、不变形。

UI结构：右侧主菜单收缩滑出；带功能强调色识别线的 UI 面板从左侧滑入，准确显示：“PLAYER 1”“RIGHT ARM EQUIPMENT”。装备列表中先高亮“PHANTOM GRIP”，随后选区移动至“CHRONOS CLAW”。

动作：PLAYER 1 的右袖口自动打开，轻量机械结构从前臂下方展开。手指分开，修长爪状指节滑入并逐一锁定，内部短暂露出精细线路、微型活塞和金属连接件。配置完成后，功能强调色 LED 依次亮起。

动态风格填充：机械结构、UI面板材质、图标、线条、锁定动画和光效根据 {visual_style} 优化，但必须轻量、精密、灵活，符合 PLAYER 1 身形，不改变脸、发型和服装主体。

声音：精密机械展开声、轻快 UI 切换音、细小锁定声。

### [4秒–7秒] — PLAYER 2 重型手臂配置
景别/机位：中景，摄影机沿两人之间平滑横移并绕向 PLAYER 2 左侧。PLAYER 1 留在背景中，轻轻观察自己完成配置的机械手。

UI结构：新的暖色/琥珀红识别 UI 滑入，准确显示：“PLAYER 2”“ARMAMENT CUSTOMIZATION”。面板以网格形式展示：
“HAND”
“FOREARM”
“ELBOW”
“UPPER ARM”
选区快速但清晰地在四个组件之间切换。

动作：PLAYER 2 左臂外套袖口分段打开，厚重前臂护板向外弹开，旧组件脱离，新型装甲沿导轨滑入；肘关节替换为厚实机械轴承，宽大的机械手重新组合并锁定。更换过程中短暂露出粗壮线路、液压活塞和深色金属骨架。每个部件锁定时亮起低调暖色指示灯。

动态风格填充：重型机械臂、UI面板、组件图标、材质、光效根据 {visual_style} 优化，但必须厚重、宽大、有力量感，与 PLAYER 1 的轻量机械爪形成清晰对比。

声音：低沉电机声、重型机械扣合声、厚重锁定反馈。

### [7秒–8.5秒] — 双人确认配置
景别/机位：中景拉回，镜头平滑回到双人构图，PLAYER 1 左侧，PLAYER 2 右侧。

UI结构：两组装备面板向画面中央汇合，形成共享按钮，准确显示：“CONFIRM CONFIG”。按钮边框、辉光、图标和贴纸质感根据 {visual_style} 优化，但层级清晰、文字可读。

动作：光标点击按钮，功能强调色能量脉冲流过 PLAYER 1 的机械爪，暖色/琥珀红能量脉冲流过 PLAYER 2 的机械拳。所有 UI 面板快速向内收缩并消失。两人同时解开交叉的双腿并调整坐姿：PLAYER 1 轻盈抬起单膝，修长机械爪依次活动手指；PLAYER 2 一只脚稳稳踩地，厚重机械拳缓慢握紧。

声音：确认提示音、双色能量脉冲、UI 收缩声。

### [8.5秒–10秒] — 双人世界加载
景别/机位：全景，底部共享加载条出现。

UI结构：加载条准确显示：“LOADING”。进度从 0% 快速填充至 100%。左半段使用 PLAYER 1 的功能强调色，右半段使用 PLAYER 2 的暖色/力量色。HUD 和加载条的形态、边框、纹理、字体根据 {visual_style} 优化，但必须清晰可读。

环境转化：黄色平面环境连续转化为游戏世界。警戒线变成真实街道道路标记和施工围挡；墨迹/纹理化为潮湿路面反光或符合 {visual_style} 的地面光影；平面涂鸦扩展成建筑墙面的喷绘；黑色背景区域形成城市阴影和深巷入口。

关键约束：转化必须连续自然，不使用硬切，不用烟雾遮挡，不改变两位角色身份。

声音：加载上升音、环境从菜单氛围过渡到城市氛围。

### [10秒–15秒] — 双人进入游戏世界
景别/机位：大全景转第三人称跟拍。加载 100% 的瞬间，两名角色同时起身。摄影机平滑下降并绕到两人身后，变成稳定第三人称双人合作视角。

世界风格：完整游戏世界根据 {visual_style} 动态生成。保留游戏开场进入世界的结构：密集建筑、道路、标志灯、招牌、人群或动态背景、快速经过的载具、电线、工业管道、远处城市天际线。具体世界材质、建筑形态、招牌设计、光影和动效必须服从 {visual_style}，不要让源默认赛博朋克风格压过用户选择，除非用户选择的是赛博朋克。

角色关系：清楚展示两人的背影和体型差异：PLAYER 1 左侧，高挑修长，轻量机械爪自然垂下；PLAYER 2 右侧，矮壮宽厚，重型机械拳微微抬起。PLAYER 1 率先迈步，PLAYER 2 紧随其后，两人并肩进入街道。

HUD结构：HUD 淡入。右上角出现小地图；左下角出现两组独立状态栏：“{player1_name}”“{player2_name}”。{player1_name} 状态栏使用 PLAYER 1 功能强调色，{player2_name} 状态栏使用 PLAYER 2 暖色/力量色。前方街道中央出现共享任务标记。

声音：城市环境氛围、远处载具声、脚步声、HUD 淡入提示音。

## Negative constraints
No third player, no female character unless explicitly requested by user, no duplicated character, no character swapping, no username swapping, no merged bodies, no identical body proportions, no changing faces, no changing hairstyles, no changing identity, no missing PLAYER 1, no missing PLAYER 2, no split screen, no hard cuts, no random camera shake, no floating body parts, no gruesome dismemberment, no weapon, no excessive neon unless selected by user style, no purple background unless selected by user style, no unreadable UI, no random letters, no misspelled usernames, no extra menu options, no official game logo, no copied branded interface, no watermark.
