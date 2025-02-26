import { useEffect, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { CONFIG_DEFAULT } from '../Config';
import { isDev } from '../Config';
import StorageUtils from '../utils/storage';
import { isBoolean, isNumeric, isString } from '../utils/misc';
import {
  BeakerIcon,
  ChatBubbleOvalLeftEllipsisIcon,
  Cog6ToothIcon,
  FunnelIcon,
  HandRaisedIcon,
  SquaresPlusIcon,
} from '@heroicons/react/24/outline';
import { OpenInNewTab } from '../utils/common';
import { useTranslation } from 'react-i18next';
import { HeaderLanguageBlock, HeaderThemeBlock } from './Header.tsx';

type SettKey = keyof typeof CONFIG_DEFAULT;

export default function SettingDialog() {
  const { t } = useTranslation();
  const { config, saveConfig } = useAppContext();

  const BASIC_KEYS: SettKey[] = [
    'temperature',
    'top_k',
    'top_p',
    'min_p',
    'max_tokens',
  ];
  const SAMPLER_KEYS: SettKey[] = [
    'dynatemp_range',
    'dynatemp_exponent',
    'typical_p',
    'xtc_probability',
    'xtc_threshold',
  ];
  const PENALTY_KEYS: SettKey[] = [
    'repeat_last_n',
    'repeat_penalty',
    'presence_penalty',
    'frequency_penalty',
    'dry_multiplier',
    'dry_base',
    'dry_allowed_length',
    'dry_penalty_last_n',
  ];

  enum SettingInputType {
    SHORT_INPUT,
    LONG_INPUT,
    CHECKBOX,
    CUSTOM,
  }

  interface SettingFieldInput {
    type: Exclude<SettingInputType, SettingInputType.CUSTOM>;
    label: string | React.ReactElement;
    help?: string | React.ReactElement;
    key: SettKey;
  }

  interface SettingFieldCustom {
    type: SettingInputType.CUSTOM;
    key: SettKey;
    component:
      | string
      | React.FC<{
          value: string | boolean | number | never[] | string[];
          onChange: (value: string) => void;
        }>;
  }

  interface SettingSection {
    title: React.ReactElement;
    fields: (SettingFieldInput | SettingFieldCustom)[];
  }

  const ICON_CLASSNAME = 'w-4 h-4 mr-1 inline';

  const SETTING_SECTIONS: SettingSection[] = [
    {
      title: (
        <>
          <Cog6ToothIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.General')}
        </>
      ),
      fields: [
        {
          type: SettingInputType.LONG_INPUT,
          label: t('Settings.labels.systemMessage'),
          key: 'systemMessage',
        },
        ...BASIC_KEYS.map(
          (key) =>
            ({
              type: SettingInputType.SHORT_INPUT,
              label: key,
              key,
            }) as SettingFieldInput
        ),
        {
          type: SettingInputType.SHORT_INPUT,
          label: t('Settings.labels.apiKey'),
          key: 'apiKey',
        },
      ],
    },
    {
      title: (
        <>
          <FunnelIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.Samplers')}
        </>
      ),
      fields: [
        {
          type: SettingInputType.SHORT_INPUT,
          label: t('Settings.labels.samplers'),
          key: 'samplers',
        },
        ...SAMPLER_KEYS.map(
          (key) =>
            ({
              type: SettingInputType.SHORT_INPUT,
              label: key,
              key,
            }) as SettingFieldInput
        ),
      ],
    },
    {
      title: (
        <>
          <HandRaisedIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.Penalties')}
        </>
      ),
      fields: PENALTY_KEYS.map((key) => ({
        type: SettingInputType.SHORT_INPUT,
        label: key,
        key,
      })),
    },
    {
      title: (
        <>
          <ChatBubbleOvalLeftEllipsisIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.Reasoning')}
        </>
      ),
      fields: [
        {
          type: SettingInputType.CHECKBOX,
          label: t('Settings.labels.showThoughtInProgress'),
          key: 'showThoughtInProgress',
        },
        {
          type: SettingInputType.CHECKBOX,
          label: t('Settings.labels.excludeThoughtOnReq'),
          key: 'excludeThoughtOnReq',
        },
      ],
    },
    {
      title: (
        <>
          <SquaresPlusIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.Advanced')}
        </>
      ),
      fields: [
        {
          type: SettingInputType.CUSTOM,
          key: 'custom', // dummy key, won't be used
          component: () => {
            const debugImportDemoConv = async () => {
              const res = await fetch('/demo-conversation.json');
              const demoConv = await res.json();
              StorageUtils.remove(demoConv.id);
              for (const msg of demoConv.messages) {
                StorageUtils.appendMsg(demoConv.id, msg);
              }
            };
            return (
              <button className="btn" onClick={debugImportDemoConv}>
                {t('Settings.labels.customBtn')}
              </button>
            );
          },
        },
        {
          type: SettingInputType.CHECKBOX,
          label: t('Settings.labels.showTokensPerSecond'),
          key: 'showTokensPerSecond',
        },
        {
          type: SettingInputType.LONG_INPUT,
          label: (
            <>
              {t('Settings.labels.custom')}{' '}
              <OpenInNewTab href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">
                {t('Settings.labels.customLinkLabel')}
              </OpenInNewTab>
              ){' '}
            </>
          ),
          key: 'custom',
        },
      ],
    },
    {
      title: (
        <>
          <BeakerIcon className={ICON_CLASSNAME} />
          {t('Settings.sections.Experimental')}
        </>
      ),
      fields: [
        {
          type: SettingInputType.CUSTOM,
          key: 'custom', // dummy key, won't be used
          component: () => (
            <>
              <p className="mb-8">
                {t('Settings.labels.Experimental1')}
                <br />
                <br />
                {t('Settings.labels.Experimental2')}{' '}
                <OpenInNewTab href="https://github.com/ggerganov/llama.cpp/issues/new?template=019-bug-misc.yml">
                  Bug (misc.)
                </OpenInNewTab>{' '}
                {t('Settings.labels.Experimental3')}
                <br />
                <br />
                {t('Settings.labels.Experimental4')}
              </p>
            </>
          ),
        },
        {
          type: SettingInputType.CHECKBOX,
          label: (
            <>
              <b>{t('Settings.labels.pyIntepreter1')}</b>
              <br />
              <small className="text-xs">
                {t('Settings.labels.pyIntepreter2')}{' '}
                <OpenInNewTab href="https://pyodide.org">pyodide</OpenInNewTab>,
                {t('Settings.labels.pyIntepreter3')}
              </small>
            </>
          ),
          key: 'pyIntepreterEnabled',
        },
      ],
    },
  ];

  const {
    closeDropDownMenu,
    promptSelectOptions,
    promptSelectConfig,
    resetSettings,
  } = useAppContext();
  const [selectedConfig, setSelectedConfig] = useState<number>(-1);
  // const [sectionIdx, setSectionIdx] = useState(0);
  // const { closeDropDownMenu } = useAppContext();

  // clone the config object to prevent direct mutation
  const [localConfig, setLocalConfig] = useState<typeof CONFIG_DEFAULT>(
    JSON.parse(JSON.stringify(config))
  );
  // when config is changed, reload localconfig
  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const resetConfig = () => {
    if (window.confirm(t('Settings.resetConfirm'))) {
      setLocalConfig(CONFIG_DEFAULT);
    }
  };

  const handleSave = () => {
    // copy the local config to prevent direct mutation
    const newConfig: typeof CONFIG_DEFAULT = JSON.parse(
      JSON.stringify(localConfig)
    );
    // validate the config
    for (const key in newConfig) {
      const value = localConfig[key as SettKey];
      const mustBeBoolean = isBoolean(CONFIG_DEFAULT[key as SettKey]);
      const mustBeString = isString(CONFIG_DEFAULT[key as SettKey]);
      const mustBeNumeric = isNumeric(CONFIG_DEFAULT[key as SettKey]);
      const mustBeArray = Array.isArray(CONFIG_DEFAULT[key as SettKey]);
      if (mustBeString) {
        if (!isString(value)) {
          alert(
            `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave2')}`
          );
          return;
        }
      } else if (mustBeNumeric) {
        const trimedValue = value.toString().trim();
        const numVal = Number(trimedValue);
        if (isNaN(numVal) || !isNumeric(numVal) || trimedValue.length === 0) {
          alert(
            `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave3')}`
          );
          return;
        }
        // force conversion to number
        // @ts-expect-error this is safe
        newConfig[key] = numVal;
      } else if (mustBeBoolean) {
        if (!isBoolean(value)) {
          alert(
            `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave4')}`
          );
          return;
        }
      } else if (mustBeArray) {
        if (!Array.isArray(value)) {
          alert(
            `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave5')}`
          );
          return;
        }
      } else {
        console.error(`Unknown default type for key ${key}`);
      }
    }
    if (isDev) console.log('Saving config', newConfig);
    saveConfig(newConfig);
  };

  const onChange = (key: SettKey) => (value: string | boolean) => {
    // note: we do not perform validation here, because we may get incomplete value as user is still typing it
    setLocalConfig({ ...localConfig, [key]: value });
  };
  const selectPrompt = (value: number) => {
    setSelectedConfig(value);
    if (value === -1) {
      resetSettings();
      return;
    }
    if (
      promptSelectConfig &&
      promptSelectConfig[value] &&
      promptSelectConfig[value].config
    ) {
      const newConfig: typeof CONFIG_DEFAULT = JSON.parse(
        JSON.stringify(CONFIG_DEFAULT)
      );
      // validate the config
      for (const key in promptSelectConfig[value].config) {
        const val =
          promptSelectConfig[value].config[key as keyof typeof CONFIG_DEFAULT];
        const mustBeBoolean = isBoolean(
          CONFIG_DEFAULT[key as keyof typeof CONFIG_DEFAULT]
        );
        const mustBeString = isString(
          CONFIG_DEFAULT[key as keyof typeof CONFIG_DEFAULT]
        );
        const mustBeNumeric = isNumeric(
          CONFIG_DEFAULT[key as keyof typeof CONFIG_DEFAULT]
        );
        const mustBeArray = Array.isArray(
          CONFIG_DEFAULT[key as keyof typeof CONFIG_DEFAULT]
        );
        if (mustBeString) {
          if (!isString(val)) {
            console.log(
              `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave2')}`
            );
            console.log(value);
            return;
          }
        } else if (mustBeNumeric) {
          const trimedValue = val.toString().trim();
          const numVal = Number(trimedValue);
          if (isNaN(numVal) || !isNumeric(numVal) || trimedValue.length === 0) {
            console.log(
              `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave3')}`
            );
            console.log(value);
            return;
          }
          // force conversion to number
          // @ts-expect-error this is safe
          newConfig[key] = numVal;
        } else if (mustBeBoolean) {
          if (!isBoolean(val)) {
            console.log(
              `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave4')}`
            );
            console.log(value);
            return;
          }
        } else if (mustBeArray) {
          if (!Array.isArray(val)) {
            console.log(
              `${t('Settings.labels.handleSave1')} ${key} ${t('Settings.labels.handleSave5')}`
            );
            console.log(val);
            return;
          }
        } else {
          console.error(`Unknown default type for key ${key}`);
          console.log(value);
        }
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        newConfig[key] = val;
      }
      if (isDev) console.log('Saving config', newConfig);
      saveConfig(CONFIG_DEFAULT);
      saveConfig(newConfig);
      resetSettings();
    }
  };
  const downloadConfigs = () => {
    const configJson = JSON.stringify({ presets: promptSelectConfig }, null, 2);
    const blob = new Blob([configJson], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `config.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const { setPromptSelectConfig, settingsSeed } = useAppContext();
  const onFileChange = () => {
    const inputE: HTMLInputElement = document?.getElementById(
      'configJsonInput'
    ) as HTMLInputElement;
    let files: FileList | null = null;
    if (inputE && inputE.files) {
      files = inputE.files;
    } else {
      return;
    }
    if (files.length <= 0) {
      return false;
    }
    const fr = new FileReader();
    fr.onload = function (e) {
      const result = JSON.parse(e?.target?.result as string);
      if (result && result.presets) {
        setPromptSelectConfig(result.presets, () => {
          resetSettings();
        });
      }
      // const formatted = JSON.stringify(result, null, 2);
      console.log('JSON loaded !');
      resetSettings();
    };
    const fItem: Blob | null = files.item(0);
    if (fItem) {
      fr.readAsText(fItem);
    }
    return;
  };

  return (
    <div
      key={settingsSeed}
      className="h-screen overflow-y-auto overflow-x-clip flex flex-col bg-base-200 py-4 px-4"
    >
      <div className="flex flex-row items-center justify-between mt-4 absolute top-0 right-0 z-50">
        <div
          className="tooltip tooltip-bottom z-100"
          data-tip={t('Settings.CloseBtn')}
          onClick={() => {
            const elem = document.getElementById('settingBlock');
            const elem2 = document.getElementById('mainBlock');
            if (elem && elem2) {
              elem.style.display = 'none';
              elem2.style.display = 'block';
            }
          }}
        >
          <button className="btn">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="size-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
              />
            </svg>
          </button>
        </div>
      </div>
      <div className="flex flex-row items-center justify-between mt-4 z-10">
        <h2 className="font-bold ml-4">{t('Settings.Settings')}</h2>
      </div>
      <div className="text-right block lg:hidden">
        <div className="">
          <HeaderThemeBlock id="theme-dropdown-2" />
          <HeaderLanguageBlock id="language-dropdown-2" />
        </div>
      </div>
      <div className="inline">
        <div className="px-4 mt-4 flex">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="size-5"
          >
            <path d="M10 3.75a2 2 0 1 0-4 0 2 2 0 0 0 4 0ZM17.25 4.5a.75.75 0 0 0 0-1.5h-5.5a.75.75 0 0 0 0 1.5h5.5ZM5 3.75a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1 0-1.5h1.5a.75.75 0 0 1 .75.75ZM4.25 17a.75.75 0 0 0 0-1.5h-1.5a.75.75 0 0 0 0 1.5h1.5ZM17.25 17a.75.75 0 0 0 0-1.5h-5.5a.75.75 0 0 0 0 1.5h5.5ZM9 10a.75.75 0 0 1-.75.75h-5.5a.75.75 0 0 1 0-1.5h5.5A.75.75 0 0 1 9 10ZM17.25 10.75a.75.75 0 0 0 0-1.5h-1.5a.75.75 0 0 0 0 1.5h1.5ZM14 10a2 2 0 1 0-4 0 2 2 0 0 0 4 0ZM10 16.25a2 2 0 1 0-4 0 2 2 0 0 0 4 0Z" />
          </svg>
          <div>{t('Settings.presetLabel')}</div>
        </div>
        <div className="flex justify-end">
          <div
            className="tooltip tooltip-bottom z-100"
            data-tip={t('Settings.loadPresetBtn')}
            onClick={() => {
              document?.getElementById('configJsonInput')?.click();
            }}
          >
            <input
              id="configJsonInput"
              className="hidden"
              type="file"
              onChange={() => {
                onFileChange();
              }}
              accept=".json"
            />
            <div className="dropdown dropdown-end dropdown-bottom">
              <div tabIndex={0} role="button" className="btn m-1">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="size-6"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15M9 12l3 3m0 0 3-3m-3 3V2.25"
                  />
                </svg>
              </div>
            </div>
          </div>
          <div
            className="tooltip tooltip-bottom z-100"
            data-tip={t('Settings.savePresetBtn')}
            onClick={downloadConfigs}
          >
            <div className="dropdown dropdown-end dropdown-bottom">
              <div tabIndex={0} role="button" className="btn m-1">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width={24}
                  height={24}
                  viewBox="0 0 24 24"
                >
                  <path
                    fill="currentColor"
                    d="M20 7.423v10.962q0 .69-.462 1.153T18.384 20H5.616q-.691 0-1.153-.462T4 18.384V5.616q0-.691.463-1.153T5.616 4h10.961zm-1 .427L16.15 5H5.616q-.27 0-.443.173T5 5.616v12.769q0 .269.173.442t.443.173h12.769q.269 0 .442-.173t.173-.443zm-7 8.688q.827 0 1.414-.586T14 14.538t-.587-1.413T12 12.539t-1.413.586T10 14.538t.587 1.414t1.413.586M6.77 9.77h7.422v-3H6.77zM5 7.85V19V5z"
                  ></path>
                </svg>
              </div>
            </div>
          </div>
          {promptSelectOptions.length > 0 ? (
            <div
              className="tooltip tooltip-bottom z-100"
              data-tip={t('Settings.tooltipPresets')}
            >
              <div className="dropdown dropdown-end dropdown-bottom">
                <div tabIndex={0} role="button" className="btn m-1">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={1.5}
                    stroke="currentColor"
                    className="size-6"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"
                    />
                  </svg>
                </div>
                <ul
                  tabIndex={0}
                  className="dropdown-content bg-base-300 rounded-box z-[1] w-52 p-2 shadow-2xl h-80 overflow-y-auto"
                >
                  {[...promptSelectOptions].map((opt) => (
                    <li key={opt.key}>
                      <input
                        type="radio"
                        name="settings"
                        className="theme-controller btn btn-sm btn-block btn-ghost justify-start"
                        aria-label={opt.value}
                        value={opt.value}
                        checked={selectedConfig === opt.key}
                        onChange={(e) =>
                          e.target.checked && selectPrompt(opt.key)
                        }
                        onClick={() => {
                          closeDropDownMenu('');
                        }}
                      />
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <div className="flex flex-col">
        {/* Right panel, showing setting fields */}
        <div className="grow px-4">
          {SETTING_SECTIONS.map((section, idx) => (
            <div key={idx}>
              <div className="pt-3 pb-1">{section.title}</div>
              {section.fields.map((field, sIdx) => {
                const key = `${idx}-${sIdx}-${field.key}-${section.title}`;
                if (field.type === SettingInputType.SHORT_INPUT) {
                  return (
                    <SettingsModalShortInput
                      key={key}
                      configKey={field.key}
                      value={localConfig[field.key]}
                      onChange={onChange(field.key)}
                      label={field.label as string}
                    />
                  );
                } else if (field.type === SettingInputType.LONG_INPUT) {
                  return (
                    <SettingsModalLongInput
                      key={key}
                      configKey={field.key}
                      value={localConfig[field.key].toString()}
                      onChange={onChange(field.key)}
                      label={field.label as string}
                    />
                  );
                } else if (field.type === SettingInputType.CHECKBOX) {
                  return (
                    <SettingsModalCheckbox
                      key={key}
                      configKey={field.key}
                      value={!!localConfig[field.key]}
                      onChange={onChange(field.key)}
                      label={field.label as string}
                    />
                  );
                } else if (field.type === SettingInputType.CUSTOM) {
                  return (
                    <div key={key} className="mb-2">
                      {typeof field.component === 'string'
                        ? field.component
                        : field.component({
                            value: localConfig[field.key],
                            onChange: onChange(field.key),
                          })}
                    </div>
                  );
                }
              })}
            </div>
          ))}
          <p className="opacity-40 mb-6 text-sm mt-8">
            {t('Settings.savedLocal')}
          </p>
        </div>
      </div>
      <div className="flex flex-row items-center justify-between mt-4 sticky bottom-0 z-10">
        <div
          className="tooltip tooltip-top z-100"
          data-tip={t('Settings.resetBtn')}
          onClick={() => {
            resetConfig();
          }}
        >
          <button className="btn">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="size-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19.5 12c0-1.232-.046-2.453-.138-3.662a4.006 4.006 0 0 0-3.7-3.7 48.678 48.678 0 0 0-7.324 0 4.006 4.006 0 0 0-3.7 3.7c-.017.22-.032.441-.046.662M19.5 12l3-3m-3 3-3-3m-12 3c0 1.232.046 2.453.138 3.662a4.006 4.006 0 0 0 3.7 3.7 48.656 48.656 0 0 0 7.324 0 4.006 4.006 0 0 0 3.7-3.7c.017-.22.032-.441.046-.662M4.5 12l3 3m-3-3-3 3"
              />
            </svg>
          </button>
        </div>
        <div
          className="tooltip tooltip-top z-100"
          data-tip={t('Settings.saveBtn')}
          onClick={() => {
            handleSave();
          }}
        >
          <button className="btn btn-primary">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width={24}
              height={24}
              viewBox="0 0 24 24"
            >
              <path
                fill="currentColor"
                d="M20 7.423v10.962q0 .69-.462 1.153T18.384 20H5.616q-.691 0-1.153-.462T4 18.384V5.616q0-.691.463-1.153T5.616 4h10.961zm-1 .427L16.15 5H5.616q-.27 0-.443.173T5 5.616v12.769q0 .269.173.442t.443.173h12.769q.269 0 .442-.173t.173-.443zm-7 8.688q.827 0 1.414-.586T14 14.538t-.587-1.413T12 12.539t-1.413.586T10 14.538t.587 1.414t1.413.586M6.77 9.77h7.422v-3H6.77zM5 7.85V19V5z"
              ></path>
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function SettingsModalLongInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: string;
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <label className="form-control mb-2">
      <div className="label inline">{label || configKey}</div>
      <textarea
        className="textarea textarea-bordered h-24"
        placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function SettingsModalShortInput({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any;
  onChange: (value: string) => void;
  label?: string;
}) {
  const { t } = useTranslation();
  const transHelpMsg = t('Settings.meaning.' + configKey);
  return (
    <>
      {/* on mobile, we simply show the help message here */}
      {transHelpMsg && (
        <div className="block md:hidden mb-1">
          <b>{label || configKey}</b>
          <br />
          <p className="text-xs">{transHelpMsg}</p>
        </div>
      )}
      <label className="input input-bordered join-item grow flex items-center gap-2 mb-2">
        <div className="dropdown dropdown-hover">
          <div tabIndex={0} role="button" className="font-bold hidden md:block">
            {label || configKey}
          </div>
          {transHelpMsg && (
            <div className="dropdown-content menu bg-base-100 rounded-box z-10 w-64 p-2 shadow mt-4">
              {transHelpMsg}
            </div>
          )}
        </div>
        <input
          type="text"
          className="grow"
          placeholder={`Default: ${CONFIG_DEFAULT[configKey] || 'none'}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </>
  );
}

function SettingsModalCheckbox({
  configKey,
  value,
  onChange,
  label,
}: {
  configKey: SettKey;
  value: boolean;
  onChange: (value: boolean) => void;
  label: string;
}) {
  return (
    <div className="flex flex-row items-center mb-2">
      <input
        type="checkbox"
        className="toggle"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="ml-4">{label || configKey}</span>
    </div>
  );
}
